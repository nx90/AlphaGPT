# Alpha Jungle: MCTS-Driven Alpha Mining

This module implements an automated alpha factor mining system using Monte Carlo Tree Search (MCTS) combined with an LLM (DeepSeek/Azure OpenAI).

## Overview

The system iterates through:
1.  **Hypothesis Generation**: LLM proposes initial alpha formulas (e.g., `-1 * ts_delta(close, 5)`).
2.  **Execution & Backtesting**: Formulas are executed on A-Share data (using PyTorch-accelerated operators) and backtested for Sharpe, IC, etc.
3.  **Refinement Loop (MCTS)**:
    - Factors are treated as nodes in a search tree.
    - The LLM acts as the "policy" to suggest refinements (mutations) to improve underperforming factors.
    - MCTS balances exploration (trying new variations) and exploitation (refining promising factors).

## Files

- `run_alpha_jungle.py`: The main entry point. Runs the full loop.
- `model_core/mcts_agent.py`: Implements the MCTS logic (Selection, Expansion, Simulation/Refinement, Backpropagation).
- `model_core/llm_client.py`: Handles communication with Azure OpenAI, with robust parsing for formula extraction.
- `model_core/alphagpt.py`: The executor environment (Sandbox) and data loading.
- `model_core/ops_lib.py`: PyTorch implementations of alpha operators (`ts_rank`, `ts_mean`, etc.).

## How to Run

1.  Ensure you have the A-Share CSV data in `data/csv/`.
2.  Configure your Azure OpenAI credentials in `model_core/llm_client.py` (if not already set).
3.  Run the script:

```bash
python run_alpha_jungle.py
```

## Output

The script prints the search progress and logs "New Factors" discovered.
At the end, it displays the **Top Discovered Alphas** sorted by a combined score (IC, Sharpe, Turnover).

Example Output:
```
=== Top Discovered Alphas ===
#1 [Score: 0.3570]: rank((log(volume) - ts_mean(log(volume), 40)) / ts_std(log(volume), 40))
   Metrics: ic=0.0236, rank_ic=0.2484, sharpe=0.6066 ...
```
