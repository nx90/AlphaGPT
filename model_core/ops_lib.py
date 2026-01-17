import torch

"""
AlphaGPT Operator Library (ops_lib)
Based on: "Navigating the Alpha Jungle" - Appendix E (Table 2)
Backend: PyTorch
Shape Standard: (N, T) -> (Stocks, TimeSteps)
"""

def _pad_nan(x: torch.Tensor, d: int) -> torch.Tensor:
    """Helper to maintain shape after rolling/delay operations"""
    if d <= 0: return x
    # Construct padding of NaNs
    padding = torch.full((x.shape[0], d), float('nan'), device=x.device, dtype=x.dtype)
    return padding

# ==========================================
# Unary Operators (Element-wise)
# ==========================================

def abs_(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)

def sign(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x)

def log(x: torch.Tensor) -> torch.Tensor:
    # Handle negative/zero safely -> NaN
    return torch.log(x.clamp(min=1e-8))

def power(x: torch.Tensor, e: float = 2.0) -> torch.Tensor:
    return torch.pow(x, e)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

def neg(x: torch.Tensor) -> torch.Tensor:
    return -x

def inv(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (x + 1e-6)

def sin(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)

def cos(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x)

def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

# ==========================================
# Time-Series Operators (Rolling Window)
# ==========================================

def ts_delay(x: torch.Tensor, d: int = 1) -> torch.Tensor:
    """The value of x at d days ago."""
    if d == 0: return x
    padding = _pad_nan(x, d)
    # Concatenate padding at start, remove d from end
    return torch.cat([padding, x[:, :-d]], dim=1)

def ts_delta(x: torch.Tensor, d: int = 1) -> torch.Tensor:
    """The difference between x and x at d days ago."""
    return x - ts_delay(x, d)

def ts_roc(x: torch.Tensor, d: int = 1) -> torch.Tensor:
    """Rate of change: (x - prev) / prev"""
    prev = ts_delay(x, d)
    return (x - prev) / (prev + 1e-6)

def ts_mean(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    """Rolling mean over past d days."""
    if d <= 1: return x
    # Using unfold method for logic correctness
    return _ts_apply_unfold(x, d, lambda sub: torch.nanmean(sub, dim=-1))


def ts_sum(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    return _ts_apply_unfold(x, d, lambda sub: torch.nansum(sub, dim=-1))

def ts_std(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    def _std_func(sub):
        return torch.std(sub, dim=-1, unbiased=False) # or True
    return _ts_apply_unfold(x, d, _std_func)

def ts_var(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    """Rolling variance."""
    def _var_func(sub):
        return torch.var(sub, dim=-1, unbiased=False)
    return _ts_apply_unfold(x, d, _var_func)

def ts_mid(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    """Rolling median (using PyTorch 'median' or quantile if available)."""
    # Note: torch.nanmedian returned (values, indices) in older versions, values only in new?
    # Safer to check. Usually .values for median/quantile if reduced.
    # torch.nanmedian operates on dimension.
    def _median_func(sub):
        return torch.nanmedian(sub, dim=-1).values
    return _ts_apply_unfold(x, d, _median_func)

def ts_skew(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    """Rolling skewness."""
    def _skew_func(sub):
        # sub: (N, T, d)
        mean = torch.nanmean(sub, dim=-1, keepdim=True)
        # diff
        diff = sub - mean
        # m2, m3
        m2 = torch.nanmean(diff**2, dim=-1)
        m3 = torch.nanmean(diff**3, dim=-1)
        return m3 / (m2.pow(1.5) + 1e-6)
    return _ts_apply_unfold(x, d, _skew_func)

def ts_kurt(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    """Rolling kurtosis."""
    def _kurt_func(sub):
        mean = torch.nanmean(sub, dim=-1, keepdim=True)
        diff = sub - mean
        m2 = torch.nanmean(diff**2, dim=-1)
        m4 = torch.nanmean(diff**4, dim=-1)
        return m4 / (m2.pow(2) + 1e-6)
    return _ts_apply_unfold(x, d, _kurt_func)

def ts_max(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    def _max_func(sub):
        # max returns (values, indices), need values. 
        # But sub has NaNs. max propogates NaNs usually?
        # torch.amax supports nan propagation control in newer torch, or use nan_to_num -inf
        m, _ = torch.max(torch.nan_to_num(sub, -1e9), dim=-1)
        return m
    return _ts_apply_unfold(x, d, _max_func)

def ts_min(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    def _min_func(sub):
        m, _ = torch.min(torch.nan_to_num(sub, 1e9), dim=-1)
        return m
    return _ts_apply_unfold(x, d, _min_func)

def ts_zscore(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    """Rolling z-score: (x - mean) / std"""
    m = ts_mean(x, d)
    s = ts_std(x, d)
    return (x - m) / (s + 1e-6)

def ts_rank(x: torch.Tensor, d: int = 20) -> torch.Tensor:
    """Rank of the current value relative to its past d days."""
    # sub shape: (N, T, d)
    # target is the last element: sub[:, :, -1]
    # We compare last element to all elements in window
    def _rank_func(sub):
        # sub: (N, T', d)
        latest = sub[:, :, -1].unsqueeze(-1) # (N, T', 1)
        # Compare latest with all in window
        # Count how many are <= latest
        # Handle NaNs: if latest is NaN, rank is NaN. If window has NaNs, ignore them.
        valid_mask = ~torch.isnan(sub)
        ranks = (sub < latest).sum(dim=-1).float() + 1.0 # 1-based rank
        # Normalized? Usually Alpha101 is just raw rank or (rank-1)/(count-1).
        # Let's return raw rank for now or pct rank.
        # "Relative ranking" implies percentiles usually.
        total_valid = valid_mask.sum(dim=-1).float()
        return (ranks - 1) / (total_valid - 1 + 1e-6)
        
    return _ts_apply_unfold(x, d, _rank_func)

def ts_corr(x: torch.Tensor, y: torch.Tensor, d: int = 20) -> torch.Tensor:
    """Rolling correlation between two series."""
    # Unfold both
    # Helper needed that handles two inputs
    if d <= 1: return torch.zeros_like(x)
    
    # Create windows (N, T, d)
    # Padding first
    pad = torch.full((x.shape[0], d-1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    y_pad = torch.cat([pad, y], dim=1)
    
    x_win = x_pad.unfold(dimension=1, size=d, step=1)
    y_win = y_pad.unfold(dimension=1, size=d, step=1)
    
    # Manual correlation to handle batch
    x_mean = torch.nanmean(x_win, dim=-1, keepdim=True)
    y_mean = torch.nanmean(y_win, dim=-1, keepdim=True)
    
    x_c = x_win - x_mean
    y_c = y_win - y_mean
    
    # Covariance
    cov = torch.nanmean(x_c * y_c, dim=-1)
    
    # Stds
    x_std = torch.std(x_win, dim=-1, unbiased=False)
    y_std = torch.std(y_win, dim=-1, unbiased=False)
    
    corr = cov / (x_std * y_std + 1e-8)
    return corr

def ts_autocorr(x: torch.Tensor, lag: int = 1, d: int = 20) -> torch.Tensor:
    """Rolling autocorrelation: Corr(x, Delay(x, lag), d)."""
    return ts_corr(x, ts_delay(x, lag), d)

def ts_cov(x: torch.Tensor, y: torch.Tensor, d: int = 20) -> torch.Tensor:
    """Rolling covariance."""
    if d <= 1: return torch.zeros_like(x)
    pad = torch.full((x.shape[0], d-1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    y_pad = torch.cat([pad, y], dim=1)
    x_win = x_pad.unfold(dimension=1, size=d, step=1)
    y_win = y_pad.unfold(dimension=1, size=d, step=1)
    
    x_mean = torch.nanmean(x_win, dim=-1, keepdim=True)
    y_mean = torch.nanmean(y_win, dim=-1, keepdim=True)
    x_c = x_win - x_mean
    y_c = y_win - y_mean
    return torch.nanmean(x_c * y_c, dim=-1)

def _ts_apply_unfold(x: torch.Tensor, d: int, func) -> torch.Tensor:
    """
    Generic helper using unfold.
    Memory usage: (N, T, d). For 5000 stocks, 3000 days, d=20 -> 300M floats -> 1.2GB. Acceptable.
    """
    if d <= 1: return x
    # Pad at the beginning with NaNs so that output t corresponds to window [t-d+1, t]
    padding = torch.full((x.shape[0], d-1), float('nan'), device=x.device, dtype=x.dtype)
    x_padded = torch.cat([padding, x], dim=1)
    
    # Unfold: (N, T_out, d)
    windows = x_padded.unfold(dimension=1, size=d, step=1)
    
    # Apply function
    # result shape: (N, T)
    return func(windows)

# ==========================================
# Cross-Sectional Operators
# ==========================================

def rank(x: torch.Tensor) -> torch.Tensor:
    """
    Cross-sectional rank (along stock dimension N).
    Normalized to [0, 1].
    """
    # dim=0 is stocks
    # Handle NaNs: sort pushes NaNs to end/start? 
    # torch.argsort doesn't handle NaNs consistently across versions/devices?
    # Better to mask NaNs.
    
    mask = ~torch.isnan(x)
    # Fill NaNs with -inf to rank them last (or handle separately)
    x_filled = torch.nan_to_num(x, -1e9)
    
    # Rank: argsort().argsort() gives rank indices
    ranks = x_filled.argsort(dim=0).argsort(dim=0).float()
    
    # Ideally, we want NaNs to have NaN rank
    # Count valid elements per column (time step)
    valid_count = mask.sum(dim=0).float()
    
    # Normalized rank: (rank + 1) / (count + 1) ? Or (rank) / (count - 1)
    # Let's use (rank) / (count - 1) -> 0..1
    out = ranks / (valid_count - 1 + 1e-6)
    
    # Restore NaNs
    out[~mask] = float('nan')
    return out

# ==========================================
# Binary Operators
# ==========================================

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y

def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y

def div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x / (y + 1e-6)

def gt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x > y).float()

def lt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x < y).float()

def if_else(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # GATE
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

def decayed_linear(x: torch.Tensor, d: int = 10) -> torch.Tensor:
    """
    Linear decay weighted moving average.
    Weights: d, d-1, ..., 1. Sum = d*(d+1)/2.
    """
    if d <= 1: return x
    # Weights shape: (1, 1, d)
    weights = torch.arange(d, 0, -1, device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    
    # We can use conv1d properly here
    # Flip weights because conv1d cross-correlates? 
    # Standard conv: output[t] = sum(weight[k] * input[t+k]) (usually) or t-k.
    # PyTorch conv1d: y[t] = sum(weight[k] * x[t-k]) if weight is treated as kernel?
    # Kernel: (out_channels, in_channels, kw).
    # If we want w[0]*x[t] + w[1]*x[t-1]... 
    # weight should be [1, 2, ..., d] or [d, ..., 1]?
    # "Linearly decay" usually means recent is higher.
    # So x[t] gets weight d, x[t-d+1] gets weight 1.
    # This matches the convolution kernel being [1, 2, ..., d] if we flip?
    # No, kernel is convolved.
    # Let's use unfold for clarity first, conv is an optimization.
    
    def _decay_func(sub):
        # sub: (N, T, d). 
        # sub[:,:, -1] is x[t], sub[:,:, 0] is x[t-d+1]
        # We want x[t] * d + ... + x[t-d+1] * 1
        # so weights should be [1, 2, ..., d] applied to [0, ..., d-1]
        w = torch.arange(1, d + 1, device=sub.device, dtype=sub.dtype)
        w = w / w.sum()
        # Broadcast w to (1, 1, d)
        return (sub * w).sum(dim=-1)

    return _ts_apply_unfold(x, d, _decay_func)


# ==========================================
# Legacy AlphaGPT Compatibility Operators
# ==========================================

def _ts_delay_zero(x: torch.Tensor, d: int) -> torch.Tensor:
    """Legacy AlphaGPT delay: pads zeros instead of NaNs."""
    if d <= 0:
        return x
    if d >= x.shape[1]:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-d]], dim=1)

def gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Alias for if_else (legacy name: GATE)."""
    return if_else(condition, x, y)

def jump(x: torch.Tensor) -> torch.Tensor:
    """Legacy AlphaGPT JUMP: relu(zscore(x) - 3) using per-stock global mean/std."""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

def decay(x: torch.Tensor) -> torch.Tensor:
    """Legacy AlphaGPT DECAY: x + 0.8*Delay1(x) + 0.6*Delay2(x)."""
    return x + 0.8 * _ts_delay_zero(x, 1) + 0.6 * _ts_delay_zero(x, 2)

def delay1(x: torch.Tensor) -> torch.Tensor:
    """Legacy AlphaGPT DELAY1."""
    return _ts_delay_zero(x, 1)

def max3(x: torch.Tensor) -> torch.Tensor:
    """Legacy AlphaGPT MAX3: max(x, Delay1(x), Delay2(x))."""
    return torch.max(x, torch.max(_ts_delay_zero(x, 1), _ts_delay_zero(x, 2)))

