import numpy as np


def pvi_signal(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 255,
) -> tuple:
    """Positive Volume Index with signal line.

    PVI changes only on days when volume increases from prior day.
    Signal line is EMA of PVI. PVI above signal = bullish (smart money buying).
    Returns (pvi_val, pvi_signal, pvi_above_signal).
    pvi_above_signal: 1.0 when PVI > signal, 0.0 otherwise, NaN during warmup.
    """
    n = len(closes)
    if n == 0:
        emp = np.array([], dtype=np.float64)
        return emp.copy(), emp.copy(), emp.copy()

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    # PVI: starts at 1000, updates only when volume > prior volume
    pvi_val = np.full(n, np.nan)
    pvi_val[0] = 1000.0

    for i in range(1, n):
        if volumes[i] > volumes[i - 1] and closes[i - 1] > 0:
            pct_change = (closes[i] - closes[i - 1]) / closes[i - 1]
            pvi_val[i] = pvi_val[i - 1] * (1.0 + pct_change)
        else:
            pvi_val[i] = pvi_val[i - 1]

    # Signal line: EMA of PVI
    pvi_sig = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)

    if n >= period:
        pvi_sig[period - 1] = np.mean(pvi_val[:period])
        for i in range(period, n):
            pvi_sig[i] = pvi_sig[i - 1] * (1.0 - alpha) + pvi_val[i] * alpha

    # PVI above signal
    pvi_above = np.full(n, np.nan)
    valid = (~np.isnan(pvi_val)) & (~np.isnan(pvi_sig))
    pvi_above[valid & (pvi_val > pvi_sig)] = 1.0
    pvi_above[valid & (pvi_val <= pvi_sig)] = 0.0

    return pvi_val, pvi_sig, pvi_above
