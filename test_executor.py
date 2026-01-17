from model_core.data_loader_ashare import AShareDataLoader
from model_core.executor import AlphaExecutor
import torch

def test_executor():
    print("Initializing Data Loader...")
    # Load small subset for speed
    loader = AShareDataLoader(data_dir='data/csv')
    loader.load_data(limit_stocks=50) 
    
    print("\nInitializing Executor...")
    executor = AlphaExecutor(loader)
    
    # Test Cases: Paper uses specific ops like ts_rank, ts_corr, etc.
    test_formulas = [
        "close", # Identity
        "rank(close)", # Cross-sectional rank
        "ts_mean(close, 10)", # Rolling mean
        "ts_rank(close, 10)", # Rolling rank
        "ts_corr(close, volume, 20)", # Rolling correlation (Corrected name)
        "ts_delta(close, 1)", # Delta
        "open / close", # Basic arithmetic
        "-1 * ts_rank(ts_delta(close, 7), 11)", # A typical alpha format
        "ts_skew(close, 20)", # Implemented in previous step
        "ts_kurt(close, 20)", # Implemented in previous step
        "invalid_function(close)" # Error case
    ]
    
    print(f"\nTesting {len(test_formulas)} formulas...")
    
    for formula in test_formulas:
        print(f"\n--- Formula: {formula} ---")
        res, err = executor.execute(formula)
        
        if err:
            print(f"Error (Expected if invalid): {err}")
        else:
            print(f"Success. Shape: {res.shape}")
            # Check values
            nans = torch.isnan(res).sum().item()
            print(f"NaN count: {nans} / {res.numel()}")
            print(f"Mean: {res[~torch.isnan(res)].mean().item():.4f}")

if __name__ == "__main__":
    test_executor()
