import numpy as np


def _vwema(closes: np.ndarray, volumes: np.ndarray, period: int) -> np.ndarray:
    """Volume-weighted EMA."""
    n = len(closes)
    out = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)

    # Seed with volume-weighted average of first period bars
    if n < period:
        return out
    vol_sum = np.sum(volumes[:period])
    if vol_sum > 0:
        out[period - 1] = np.sum(closes[:period] * volumes[:period]) / vol_sum
    else:
        out[period - 1] = np.mean(closes[:period])

    for i in range(period, n):
        # Weight alpha by relative volume
        vol_ratio = volumes[i] / (np.mean(volumes[max(0, i - period) : i]) + 1e-10)
        adj_alpha = alpha * min(vol_ratio, 3.0)  # cap to prevent instability
        adj_alpha = min(adj_alpha, 1.0)
        out[i] = out[i - 1] * (1.0 - adj_alpha) + closes[i] * adj_alpha

    return out


def vwmacd(
    closes: np.ndarray,
    volumes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple:
    """Volume-Weighted MACD.

    EMA calculation is weighted by volume, giving more weight to high-volume bars.
    Returns (vwmacd_line, signal_line, histogram).
    """
    n = len(closes)
    if n == 0:
        emp = np.array([], dtype=np.float64)
        return emp.copy(), emp.copy(), emp.copy()

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    fast_ema = _vwema(closes, volumes, fast)
    slow_ema = _vwema(closes, volumes, slow)

    vwmacd_line = np.full(n, np.nan)
    valid = (~np.isnan(fast_ema)) & (~np.isnan(slow_ema))
    vwmacd_line[valid] = fast_ema[valid] - slow_ema[valid]

    # Signal line: standard EMA of MACD line
    signal_line = np.full(n, np.nan)
    alpha = 2.0 / (signal + 1)
    first_valid = -1
    count = 0
    running_sum = 0.0
    for i in range(n):
        if np.isnan(vwmacd_line[i]):
            continue
        count += 1
        running_sum += vwmacd_line[i]
        if count == signal:
            signal_line[i] = running_sum / signal
            first_valid = i
            break

    if first_valid >= 0:
        for i in range(first_valid + 1, n):
            if np.isnan(vwmacd_line[i]):
                signal_line[i] = signal_line[i - 1]
            else:
                signal_line[i] = signal_line[i - 1] * (1.0 - alpha) + vwmacd_line[i] * alpha

    histogram = np.full(n, np.nan)
    valid2 = (~np.isnan(vwmacd_line)) & (~np.isnan(signal_line))
    histogram[valid2] = vwmacd_line[valid2] - signal_line[valid2]

    return vwmacd_line, signal_line, histogram
