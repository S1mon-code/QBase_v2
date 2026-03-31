import numpy as np

from indicators._utils import _ema


def macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence Divergence. Returns (macd_line, signal_line, histogram)."""
    n = closes.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty, empty

    nans = np.full(n, np.nan)
    if n < slow:
        return nans.copy(), nans.copy(), nans.copy()

    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)

    macd_line = fast_ema - slow_ema

    # Signal line is EMA of the valid portion of the MACD line
    valid_start = slow - 1
    macd_valid = macd_line[valid_start:]
    sig_ema = _ema(macd_valid, signal)

    signal_line = np.full(n, np.nan)
    signal_line[valid_start:] = sig_ema

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram
