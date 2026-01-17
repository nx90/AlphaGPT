from model_core.data_loader_ashare import AShareDataLoader
from model_core.executor import AlphaExecutor
from model_core.backtest import AlphaBacktest
import torch
import time

def test_backtest():
    print("Loading Data (Top 50 stocks)...")
    loader = AShareDataLoader(data_dir='data/csv')
    loader.load_data(limit_stocks=50)
    
    print("Executing Factors...")
    executor = AlphaExecutor(loader)
    
    # Define a few factors to test
    factors = {
        "Simple Momentum": "ts_mean(close - ts_delay(close, 5), 5)",
        "Mean Reversion": "-1 * (close - ts_mean(close, 20)) / ts_std(close, 20)",
        "Noise (Random)": "ts_delta(volume, 1) / (close + 1)", # Just noise usually
        "Invalid": "close / 0" # Produces Inf -> should be handled
    }
    
    backtester = AlphaBacktest()
    target = loader.target_ret
    
    print("\nStarting Backtest...")
    for name, formula in factors.items():
        print(f"\nEvaluating: {name} [{formula}]")
        t0 = time.time()
        
        # 1. Execute
        raw_factor, err = executor.execute(formula)
        if err:
            print(f"Execution Error: {err}")
            continue
            
        # 2. Backtest
        # Executor already handles NaNs, but let's be sure
        metrics = backtester.evaluate(raw_factor, target)
        dt = time.time() - t0
        
        print(f"Time: {dt*1000:.2f}ms")
        print(f"  IC: {metrics['ic']:.4f}")
        print(f"  RankIC: {metrics['rank_ic']:.4f}")
        print(f"  Sharpe: {metrics['sharpe']:.4f}")
        print(f"  Turnover: {metrics['turnover']:.4f}")
        print(f"  Score: {metrics['score']:.4f}")

if __name__ == "__main__":
    test_backtest()
