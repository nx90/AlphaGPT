import torch
from .config import ModelConfig

class AlphaBacktester:
    """
    Standard Alpha Factor Backtester for Stock Markets (A-Share).
    Evaluates factor performance using IC, RankIC, and Long-Short Returns.
    """
    def __init__(self, data_loader, device=None):
        self.device = device if device else ModelConfig.DEVICE
        self.loader = data_loader
        # Pre-fetch target returns if available
        if hasattr(self.loader, 'target_ret') and self.loader.target_ret is not None:
             self.target_ret = self.loader.target_ret
        else:
             self.target_ret = None

    def run(self, factor: torch.Tensor, top_k_pct=0.1):
        """Wrapper to run evaluation using loader's data."""
        if self.target_ret is None:
            if self.loader.target_ret is None:
                raise ValueError("DataLoader has no target_ret loaded.")
            self.target_ret = self.loader.target_ret
            
        return self.evaluate(factor, self.target_ret, top_k_pct)

    def evaluate(self, factor: torch.Tensor, target_ret: torch.Tensor, top_k_pct=0.1):
        """
        Evaluate a single alpha factor.
        
        Args:
            factor: (Stock, Time) float tensor, the alpha signals.
            target_ret: (Stock, Time) float tensor, usually T+1 or T+2 returns.
            top_k_pct: float, top percentile to hold (e.g., 0.1 for top 10%).

        Returns:
            metrics (dict): {
                'ic': float,
                'rank_ic': float,
                'sharpe': float,
                'turnover': float,
                'score': float
            }
        """
        # 1. Pre-check
        if factor.shape != target_ret.shape:
            raise ValueError(f"Shape mismatch: Factor {factor.shape} vs Target {target_ret.shape}")
        
        # Determine valid mask: where both factor and target are valid (non-zero/non-nan usually handled outside, 
        # but let's assume raw data 0.0 means missing)
        # In A-Share loader, we filled NaN with 0.0. A return of exact 0.0 usually means sus data or halt.
        # Let's trust the input for now but mask out strict zeros if needed.
        # For simplicity in massive tensor ops, we compute across all.
        
        eps = 1e-6

        # 2. Information Coefficient (IC) - Pearson Corr per day (cross-sectional)
        # IMPORTANT: Many generated factors can be degenerate (constant across stocks) after sanitization.
        # If cross-sectional std is ~0 on a day, IC/RankIC should be treated as invalid (0 contribution).
        f_mean = factor.mean(dim=0, keepdim=True)
        t_mean = target_ret.mean(dim=0, keepdim=True)
        f_centered = factor - f_mean
        t_centered = target_ret - t_mean

        cov = (f_centered * t_centered).mean(dim=0)
        f_std = factor.std(dim=0, unbiased=False)
        t_std = target_ret.std(dim=0, unbiased=False)

        valid_day = (f_std > eps) & (t_std > eps)

        ic_ts = cov / (f_std * t_std + 1e-9)
        ic_ts = torch.nan_to_num(ic_ts, nan=0.0, posinf=0.0, neginf=0.0)
        ic = ic_ts[valid_day].mean().item() if valid_day.any() else 0.0

        # 3. Rank IC (Spearman) - requires sorting every day
        # PyTorch argsort/rank is heavy. We can approximate or skip if speed is critical.
        # For MCTS, we might use IC as proxy for speed, or do RankIC on sampled days.
        # Let's implement full RankIC but verify performance later.
        # rank() over dim=0 (stocks)
        # Using a simple rank approximation or torch.argsort
        rank_f = factor.argsort(dim=0).argsort(dim=0).float()
        rank_t = target_ret.argsort(dim=0).argsort(dim=0).float()

        rf_center = rank_f - rank_f.mean(dim=0, keepdim=True)
        rt_center = rank_t - rank_t.mean(dim=0, keepdim=True)
        rank_cov = (rf_center * rt_center).mean(dim=0)
        rank_ic_ts = rank_cov / (rank_f.std(dim=0, unbiased=False) * rank_t.std(dim=0, unbiased=False) + 1e-9)
        rank_ic_ts = torch.nan_to_num(rank_ic_ts, nan=0.0, posinf=0.0, neginf=0.0)
        # Use the same validity mask so degenerate factors don't get spurious RankIC.
        rank_ic = rank_ic_ts[valid_day].mean().item() if valid_day.any() else 0.0

        # 4. Long-Short Returns (Top 10% - Bottom 10% or just Top 10% excess)
        # We calculate "Top Quintile Return" vs "Mean Return"
        # Since we have ranks already (rank_f):
        n_stocks = factor.shape[0]
        top_cutoff = n_stocks * (1 - top_k_pct)

        # Mask for holding positions (only on valid days)
        long_mask = (rank_f >= top_cutoff).float()
        long_mask = long_mask * valid_day.unsqueeze(0).float()
        
        # Weights: Equal weight in the top bucket
        # Daily return of the long portfolio
        long_ret_ts = (long_mask * target_ret).sum(dim=0) / (long_mask.sum(dim=0) + 1e-9)
        
        # Market mean return (equal weight entire universe)
        mkt_ret_ts = target_ret.mean(dim=0)
        
        # Excess Return time series
        excess_ret_ts = long_ret_ts - mkt_ret_ts
        excess_ret_ts = excess_ret_ts[valid_day]
        if excess_ret_ts.numel() == 0:
            annualized_ret = 0.0
            annualized_vol = 0.0
            sharpe = 0.0
        else:
            annualized_ret = excess_ret_ts.mean().item() * 252
            annualized_vol = excess_ret_ts.std(unbiased=False).item() * (252 ** 0.5)
            sharpe = annualized_ret / (annualized_vol + 1e-9)

        # 5. Turnover
        # How much does the Long mask change day-to-day?
        # Turnover = (|w_t - w_{t-1}|).sum() / 2
        # Here weights are 1/K inside mask, 0 outside. 
        # Approx turnover is just fraction of stocks changing status in the top bucket.
        mask_diff = (long_mask[:, 1:] - long_mask[:, :-1]).abs().sum(dim=0)
        bucket_size = n_stocks * top_k_pct
        turnover_ts = mask_diff / (bucket_size * 2 + 1e-9)
        valid_turn = valid_day[1:] & valid_day[:-1]
        turnover = turnover_ts[valid_turn].mean().item() if valid_turn.any() else 0.0

        # 6. Composite Score for MCTS
        # Reward = RankIC * A + Sharpe * B - Turnover * C
        # Penalize low IC heavily.
        # Tunable weights
        coverage = valid_day.float().mean().item() if factor.numel() else 0.0
        score = (rank_ic * 1.0) + (sharpe * 0.2) - (turnover * 0.1)
        # Penalize low coverage (degenerate or mostly invalid factors)
        score = score * coverage

        if not (score == score):
            score = -1.0

        # Calculate IC Information Ratios (Stability)
        if valid_day.any():
            ic_std = ic_ts[valid_day].std().item()
            rank_ic_std = rank_ic_ts[valid_day].std().item()
            ic_ir = ic / (ic_std + 1e-9)
            rank_ic_ir = rank_ic / (rank_ic_std + 1e-9)
        else:
            ic_ir = 0.0
            rank_ic_ir = 0.0

        return {
            'ic': ic,
            'rank_ic': rank_ic,
            'ic_ir': ic_ir,
            'rank_ic_ir': rank_ic_ir,
            'sharpe': sharpe,
            'ir': sharpe, # Paper definition of IR is Risk-Adjusted Excess Return, which maps to Sharpe here
            'turnover': turnover,
            'annual_ret': annualized_ret,
            'score': score
        }