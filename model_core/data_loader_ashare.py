import os
import glob
import pandas as pd
import torch
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

from .config import ModelConfig

class AShareDataLoader:
    def __init__(self, data_dir: str = 'data/csv', device=None, dtype: torch.dtype = torch.float32):
        self.data_dir = data_dir
        self.device = ModelConfig.DEVICE if device is None else device
        self.dtype = dtype
        self.raw_data_cache = None
        self.feat_tensor = None
        self.target_ret = None
        self.dates = None
        self.tickers = None

    def load_data(self, limit_stocks=None):
        print(f"Loading A-Share data from {self.data_dir}...")

        all_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        if limit_stocks:
            all_files = all_files[:limit_stocks]
        
        if not all_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        dfs = []
        print(f"Reading {len(all_files)} CSV files...")

        iterator = all_files
        if tqdm is not None:
            iterator = tqdm(all_files)

        desired_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        dtype_map = {
            'open': 'float32',
            'high': 'float32',
            'low': 'float32',
            'close': 'float32',
            'volume': 'float32',
            'amount': 'float32',
        }

        for f in iterator:
            try:
                # Filename as ticker (e.g., 'SH600000.csv' -> 'SH600000')
                ticker = os.path.basename(f).replace('.csv', '')

                df = pd.read_csv(
                    f,
                    usecols=lambda c: c in desired_cols,
                    dtype=dtype_map,
                )

                if 'date' not in df.columns:
                    raise ValueError("Missing 'date' column")

                df['date'] = pd.to_datetime(df['date'])
                df['ticker'] = ticker

                keep_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'amount']
                existing_cols = [c for c in keep_cols if c in df.columns]
                df = df[existing_cols]

                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if not dfs:
            raise ValueError("No valid data loaded")

        print("Concatenating data...")
        full_df = pd.concat(dfs, ignore_index=True)
        
        print("Pivoting data to (Time, Stock) matrices...")
        # Pivot for each field
        # Ensure dates are sorted and unified
        self.dates = sorted(full_df['date'].unique())
        self.tickers = sorted(full_df['ticker'].unique())
        
        # Set index for faster pivoting
        full_df = full_df.set_index(['date', 'ticker']).sort_index()

        def to_tensor(col):
            if col not in full_df.columns:
                print(f"Warning: Column {col} not found in data. Filling with zeros.")
                return torch.zeros((len(self.tickers), len(self.dates)), device=self.device, dtype=self.dtype)
            
            # Unstack to get (Time, Stock) then transpose to (Stock, Time)
            # This handles missing dates by introducing NaNs
            pivot = full_df[col].unstack(level='ticker').reindex(index=self.dates, columns=self.tickers)
            
            # Fill Missing Values
            # Forward fill first, then backward fill (or fill 0)
            pivot = pivot.ffill().fillna(0.0)

            # Reduce CPU memory: materialize as float32 before moving to torch
            arr = pivot.to_numpy(dtype=np.float32, copy=False).T
            return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'amount': to_tensor('amount') # specific to A-share
        }
        
        # Compute Features (State) and Target
        self._compute_features()
        self._compute_target()
        
        print(f"Data Ready.")
        print(f"Features Shape: {self.feat_tensor.shape} (Stock, Channel, Time)")
        print(f"Target Shape: {self.target_ret.shape} (Stock, Time)")

    def _compute_features(self):
        """
        Compute basic 6-channel state for RL input.
        Channels:
        0. Return (Close/PrevClose - 1)
        1. LogVolume
        2. Range ((High-Low)/Close)
        3. Close location ((Close-Low)/(High-Low))
        4. LogAmount
        5. Vwap/Close (if VWAP roughly Amount/Volume) or just Amount/Volume normalized
        """
        print("Computing features...")
        o = self.raw_data_cache['open']
        h = self.raw_data_cache['high']
        l = self.raw_data_cache['low']
        c = self.raw_data_cache['close']
        v = self.raw_data_cache['volume']
        amt = self.raw_data_cache['amount']

        # 1. Returns (robust to non-positive values)
        prev_c = torch.roll(c, 1, dims=1)
        ret = (c - prev_c) / (prev_c.abs() + 1e-6)
        
        # 2. Log Vol
        log_vol = torch.log1p(v)
        
        # 3. Normalized Range (High-Low)
        hl_range = (h - l) / (c.abs() + 1e-6)
        
        # 4. Stochastic K (Position of Close in High-Low)
        # Handle zero division
        denom = (h - l) + 1e-9
        stoch_k = (c - l) / denom
        
        # 5. Log Amount
        log_amt = torch.log1p(amt)
        
        # 6. Turnover Proxy (Amount / 1e9 or something, but robust norm handles scaling)
        # Let's use Amount/Volume = Returns ~ VWAP. 
        # Actually VWAP = Amount/Volume. 
        vwap = amt / (v + 1e-6)
        vwap_rel = vwap / (c.abs() + 1e-6)  # VWAP relative to |close|
        
        def robust_norm(t):
            # Z-Score style robust normalization per stock (dim=1)
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)

        # Stack into (Stock, Channel, Time) -> INPUT_DIM=6 needed
        self.feat_tensor = torch.stack([
            robust_norm(ret),
            robust_norm(log_vol),
            robust_norm(hl_range),
            robust_norm(stoch_k),
            robust_norm(log_amt),
            robust_norm(vwap_rel)
        ], dim=1)
        
        # Handle NaNs in features (first few days)
        self.feat_tensor = torch.nan_to_num(self.feat_tensor, nan=0.0)

    def _compute_target(self):
        """
        Compute Target Return for Training.
        Usually T+1 Open to T+2 Open return in Alpha scaling.
        Or T Close to T+1 Close.
        Standard AlphaGPT: log(Open[t+2] / Open[t+1])
        """
        print("Computing targets...")
        op = self.raw_data_cache['open']
        # T+1 Open
        t1 = torch.roll(op, -1, dims=1)
        # T+2 Open
        t2 = torch.roll(op, -2, dims=1)
        
        # Robust future return proxy (works even if prices are non-positive)
        self.target_ret = (t2 - t1) / (t1.abs() + 1e-6)
        
        # Mask last 2 steps as 0
        self.target_ret[:, -2:] = 0.0
        self.target_ret = torch.nan_to_num(self.target_ret, nan=0.0)
