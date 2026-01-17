import math
import torch
import model_core.ops_lib as ops

class AlphaExecutor:
    """
    Executes string-based Alpha factors using PyTorch operators.
    Replaces the old StackVM.
    """
    def __init__(self, data_loader):
        self.context = {}
        self._ref_shape = None
        self._ref_device = None
        self._ref_dtype = None
        self._register_data(data_loader)
        self._register_ops()

    def _register_data(self, loader):
        """Injects OHLCV tensors into the execution context."""
        if loader.raw_data_cache is None:
            raise ValueError("Data loader has no data. Call load_data() first.")
        
        # Standardize variable names for the formula (usually uppercase or specific convention)
        # Based on AlphaJungle paper or typical alpha naming: open, close, volume...
        self.context['open'] = loader.raw_data_cache['open']
        self.context['high'] = loader.raw_data_cache['high']
        self.context['low'] = loader.raw_data_cache['low']
        self.context['close'] = loader.raw_data_cache['close']
        self.context['volume'] = loader.raw_data_cache['volume']
        self.context['amount'] = loader.raw_data_cache['amount']
        
        # Add basic aliases if needed
        self.context['OPEN'] = self.context['open']
        self.context['HIGH'] = self.context['high']
        self.context['LOW'] = self.context['low']
        self.context['CLOSE'] = self.context['close']
        self.context['VOLUME'] = self.context['volume']
        self.context['AMOUNT'] = self.context['amount']
        self.context['VWAP'] = self.context['amount'] / (self.context['volume'] + 1e-6)
        # Case-insensitive support for LLM which favors lowercase 'vwap'
        self.context['vwap'] = self.context['VWAP']

        ref = self.context['close']
        self._ref_shape = tuple(ref.shape)
        self._ref_device = ref.device
        self._ref_dtype = ref.dtype

    def _register_ops(self):
        """Injects all functions from ops_lib into the context."""
        # Get all public functions from ops_lib
        for name in dir(ops):
            if not name.startswith('_'):
                obj = getattr(ops, name)
                if callable(obj):
                    self.context[name] = obj

        # Ergonomic aliases (common naming in some alpha formula styles)
        # Common LLM hallucinations map to library functions
        self.context['correlation'] = self.context.get('ts_corr')
        self.context['covariance'] = self.context.get('ts_cov')
        self.context['stddev'] = self.context.get('ts_std')
        self.context['ts_stddev'] = self.context.get('ts_std') # Frequent hallucination
        self.context['ts_std_dev'] = self.context.get('ts_std') # Frequent hallucination
        self.context['delay'] = self.context.get('ts_delay')   # Classic alpha101 name
        self.context['rank'] = self.context.get('rank')        # Ensure rank is plain
        self.context['decay_linear'] = self.context.get('ts_decay_linear')
        
    def execute(self, formula_str):
        """
        Evaluates a formula string.
        Returns: (torch.Tensor) result, (str) error_message
        """
        try:
            # Clear cache before heavy execution to reduce fragmentation
            # if torch.cuda.is_available(): torch.cuda.empty_cache()

            # Sandboxed eval: no implicit builtins (prevents import/os/etc).
            safe_builtins = {
                'abs': abs,
                'min': min,
                'max': max,
                'float': float,
                'int': int,
                'round': round,
            }
            safe_globals = {'__builtins__': safe_builtins}

            # Use ops/data context as locals.
            result = eval(formula_str, safe_globals, self.context)
            
            # Allow scalar constants; broadcast to (N, T)
            if isinstance(result, (int, float)):
                if not math.isfinite(float(result)):
                    return None, f"Result is not finite: {result}"
                result = torch.full(self._ref_shape, float(result), device=self._ref_device, dtype=self._ref_dtype)

            if not isinstance(result, torch.Tensor):
                return None, f"Result is not a Tensor, got {type(result)}"
                
            # Basic sanity check
            if result.shape != self.context['close'].shape:
                 return None, f"Shape Mismatch: {result.shape} vs {self.context['close'].shape}"

            # Sanitize numerical issues by default
            result = self.validate_tensor(result)
            return result, None
            
        except Exception as e:
            return None, str(e)

    def validate_tensor(self, tensor):
        """Checks for Inf/NaN and replaces them."""
        if tensor is None: return None
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
