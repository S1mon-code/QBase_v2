import numpy as np


def pgo(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Pretty Good Oscillator: (close - SMA(close)) / ATR.

    Normalizes the distance from the mean by volatility.
    Values > 3 suggest strong uptrend, < -3 strong downtrend.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n <= period:
        return np.full(n, np.nan)

    out = np.full(n, np.nan)

    # SMA of closes
    cs = np.cumsum(closes)
    sma = np.full(n, np.nan)
    sma[period - 1] = cs[period - 1] / period
    sma[period:] = (cs[period:] - cs[:-period]) / period

    # True range
    tr = np.full(n, np.nan)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # ATR (SMA of true range)
    tr_cs = np.cumsum(tr)
    atr = np.full(n, np.nan)
    atr[period - 1] = tr_cs[period - 1] / period
    atr[period:] = (tr_cs[period:] - tr_cs[:-period]) / period

    # PGO
    valid = period - 1
    for i in range(valid, n):
        if not np.isnan(sma[i]) and not np.isnan(atr[i]) and atr[i] > 1e-12:
            out[i] = (closes[i] - sma[i]) / atr[i]

    return out
