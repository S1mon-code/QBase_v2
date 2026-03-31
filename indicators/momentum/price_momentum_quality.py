import numpy as np


def pmq(closes: np.ndarray, period: int = 20) -> np.ndarray:
    """Price Momentum Quality: momentum / volatility of momentum.

    Measures the quality (consistency) of momentum by dividing the
    raw momentum by its own standard deviation over the lookback window.
    Higher values indicate smoother, more reliable momentum.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n <= period:
        return np.full(n, np.nan)

    out = np.full(n, np.nan)
    # Raw momentum: close[i] - close[i - period]
    mom = np.full(n, np.nan)
    mom[period:] = closes[period:] - closes[:-period]

    # Rolling std of momentum over 'period' bars
    for i in range(2 * period, n):
        window = mom[i - period + 1 : i + 1]
        std = np.std(window, ddof=1)
        if std > 1e-12:
            out[i] = mom[i] / std
        else:
            out[i] = 0.0

    return out
