import sys
import torch
import traceback
import signal
import argparse

from model_core.data_loader_ashare import AShareDataLoader
from model_core.executor import AlphaExecutor
from model_core.backtest import AlphaBacktester
from model_core.llm_client import LLMClient
from model_core.mcts_agent import MCTSAgent

def main():
    print("Initializing AlphaJungle Framework...")

    parser = argparse.ArgumentParser(description="Run AlphaJungle MCTS alpha mining loop")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Torch device for tensors")
    parser.add_argument("--limit-stocks", type=int, default=0, help="Number of stocks to load (0 = all)")
    parser.add_argument("--iterations", type=int, default=50, help="MCTS iterations")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device=cuda requested but CUDA not available; falling back to cpu")
        device = "cpu"

    # 1. Load Data
    # Only load a subset of stocks for faster iteration during dev/demo
    loader = AShareDataLoader(data_dir='data/csv', device=device)
    try:
        limit = None if (args.limit_stocks is None or args.limit_stocks <= 0) else args.limit_stocks
        loader.load_data(limit_stocks=limit)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # Device diagnostics (helps explain low GPU utilization if tensors are small)
    close = loader.raw_data_cache.get("close") if loader.raw_data_cache else None
    if close is not None:
        print(f"Device Diagnostics: torch.cuda.is_available()={torch.cuda.is_available()}, selected_device={device}")
        print(f"Device Diagnostics: close.device={close.device}, close.shape={tuple(close.shape)}, close.dtype={close.dtype}")
        if close.is_cuda:
            # Note: utilization can still appear ~0% for small tensors and LLM/network waits.
            print(f"Device Diagnostics: cuda_device_name={torch.cuda.get_device_name(close.device)}")

    # 2. Components
    executor = AlphaExecutor(loader)
    backtester = AlphaBacktester(loader) # Assuming default reward args
    llm = LLMClient()
    
    # 3. Agent
    agent = MCTSAgent(
        data_loader=loader,
        llm_client=llm,
        executor=executor,
        backtester=backtester,
        exploration_weight=1.5
    )
    
    # 4. Bootstrap
    print("\nBootstrapping MCTS with initial hypotheses...")
    initial_factors = [
        "ts_rank(close, 10)", 
        "rank(volume / ts_mean(volume, 20))",
        "-1 * ts_delta(close, 5)"
    ]
    
    agent.initialize(initial_factors)
    print(f"Initialized with {len(initial_factors)} factors.")
    
    # 5. Run Loop
    try:
        agent.search(n_iterations=args.iterations)
    except KeyboardInterrupt:
        print("Search interrupted by user.")
    except Exception as e:
        traceback.print_exc()
        
    # 6. Report
    print("\n\n=== Top Discovered Alphas ===")
    top_alphas = agent.zoo.get_top_k(10)
    for i, alpha in enumerate(top_alphas):
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in alpha.metrics.items() if isinstance(v, (int, float))])
        print(f"#{i+1} [Score: {alpha.score:.4f}]: {alpha.formula}")
        print(f"   Metrics: {metrics_str}")
        print("-" * 50)

if __name__ == "__main__":
    main()
