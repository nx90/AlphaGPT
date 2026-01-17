import math
import numpy as np
import collections
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import torch

from model_core.llm_client import LLMClient
from model_core.executor import AlphaExecutor
from model_core.backtest import AlphaBacktester

@dataclass
class AlphaNode:
    formula: str
    parent: Optional['AlphaNode'] = None
    children: List['AlphaNode'] = field(default_factory=list)
    
    # MCTS Stats
    visits: int = 0
    q_value: float = 0.0  # Max reward in subtree
    
    # Performance Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    
    # Lineage for LLM context
    refinement_reason: str = "Initial"
    
    def is_root(self):
        return self.parent is None

class AlphaZoo:
    """Stores high-quality alphas found during search."""
    def __init__(self):
        self.alphas = [] # List of nodes

    def add(self, node: AlphaNode):
        self.alphas.append(node)
        # Sort by score descending
        self.alphas.sort(key=lambda x: x.score, reverse=True)
        
    def get_top_k(self, k=5):
        return self.alphas[:k]

class MCTSAgent:
    def __init__(self, 
                 data_loader, 
                 llm_client: LLMClient, 
                 executor: AlphaExecutor, 
                 backtester: AlphaBacktester,
                 exploration_weight: float = 1.414):
        
        self.data_loader = data_loader
        self.llm_client = llm_client
        self.executor = executor
        self.backtester = backtester
        self.c_puct = exploration_weight
        
        self.zoo = AlphaZoo()
        self.root = None
        self.visited_formulas = set()

    def initialize(self, initial_factors: List[str]):
        """Bootstraps the tree with initial hypotheses."""
        self.root = AlphaNode(formula="ROOT")
        self.root.visits = 1
        
        for f_str in initial_factors:
            child = AlphaNode(formula=f_str, parent=self.root)
            self._evaluate_node(child)
            self.root.children.append(child)
            self.zoo.add(child)
            # Add normalized formula to visited set
            self.visited_formulas.add(f_str.replace(" ", ""))

    def search(self, n_iterations=10):
        """Main MCTS loop."""
        print(f"Starting MCTS Search for {n_iterations} iterations...")
        
        for i in range(n_iterations):
            # 1. Selection
            leaf = self._select(self.root)
            
            # 2. Expansion & Simulation
            child = self._expand(leaf)
            
            # 3. Backpropagation
            if child:
                reward = child.score
                self._backpropagate(child, reward)
                print(f"Iter {i+1}: New Factor '{child.formula}' | Score: {child.score:.4f}")
            else:
                # If expansion failed (duplicate/LLM error), we just continue
                # print(f"Iter {i+1}: Skipped (Duplicate/Error)")
                pass

    def _select(self, node: AlphaNode) -> AlphaNode:
        """Selects a node to expand based on PUCB."""
        current = node
        
        while True:
            # If current node has no children, we must expand it
            if not current.children:
                return current
            
            # Calculates UCB for all children
            best_score = -float('inf')
            best_child = None
            
            # Candidates: existing children or expanding here?
            # For simplicity, we just traverse to the best child to drill deep.
            
            scores = []
            for child in current.children:
                try:
                    u = self.c_puct * math.sqrt(math.log(current.visits + 1) / (child.visits + 1))
                except ValueError:
                    u = 1.0
                
                # Q-value is max reward in subtree
                score = child.q_value + u
                scores.append((score, child))
            
            if not scores:
                 return current
                 
            # Find best child
            scores.sort(key=lambda x: x[0], reverse=True)
            best_child = scores[0][1]
            
            current = best_child

    def _expand(self, node: AlphaNode) -> Optional[AlphaNode]:
        """Uses LLM to generate a new child factor."""
        # 1. Analyze node weakness
        feedback = self._analyze_weakness(node)
        
        # 2. Call LLM
        if node.is_root():
             new_formulas = self.llm_client.generate_initial_hypothesis(1)
             formula_str = new_formulas[0] if new_formulas else None
        else:
             formula_str = self.llm_client.refine_factor(node.formula, feedback['reason'], node.metrics)
             
        if not formula_str:
            return None
            
        # 3. Duplicate Check
        clean_formula = formula_str.replace(" ", "")
        if clean_formula in self.visited_formulas:
            # print(f"Duplicate Factor Skip: {formula_str}")
            return None
            
        # 4. Create Child Node
        child = AlphaNode(formula=formula_str, parent=node, refinement_reason=feedback['reason'])
        
        # 5. Evaluate
        self._evaluate_node(child)
        
        # If evaluation was valid (no error), add it
        # If error, we still add to tree but with low score to record failure
        self.visited_formulas.add(clean_formula)
        node.children.append(child)
        
        if child.score > 0.05:
            self.zoo.add(child)
            
        return child

    def _evaluate_node(self, node: AlphaNode):
        """Runs the backtest and updates node metrics."""
        if node.formula == "ROOT":
            node.score = 0
            return

        try:
            # Clear CUDA cache before heavy allocs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Use no_grad to reduce memory usage for inference/search
            with torch.no_grad():
                factor_tensor, error = self.executor.execute(node.formula)
            
            if error:
                 print(f"Exec Error {node.formula}: {error}")
                 node.score = -1.0
                 node.q_value = -1.0
                 node.metrics = {'error': error}
                 return

            # 2. Add to Backtester
            result = self.backtester.run(factor_tensor)
            
            node.metrics = result
            node.score = result['score']  # Composite score from backtester
            
            # Update q_value initialized to score
            node.q_value = node.score
            
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg:
                 print(f"OOM Error for {node.formula}. Cleared cache.")
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
            else:
                 print(f"Eval Error for {node.formula}: {e}")
                 
            node.score = -1.0
            node.q_value = -1.0
            node.metrics = {'error': str(e)}

    def _backpropagate(self, node: AlphaNode, reward: float):
        """Updates Q-values up the tree."""
        current = node
        while current:
            current.visits += 1
            if reward > current.q_value:
                current.q_value = reward
            current = current.parent

    def _analyze_weakness(self, node: AlphaNode) -> Dict:
        """Heuristic to pick what to improve."""
        if not node.metrics or 'sharpe' not in node.metrics:
            return {'reason': "maximize_signal"}

        m = node.metrics
        reason = "improve_general"

        if m.get('rank_ic', 0) < 0.03:
            reason = "increase_rank_ic_correlation"
        elif m.get('turnover', 0) > 0.4:
            reason = "reduce_turnover_transaction_costs"
        elif m.get('sharpe', 0) < 1.0:
            reason = "improve_sharpe_ratio_stability"

        return {'reason': reason}
