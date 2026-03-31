import numpy as np


def er_bands(
    closes: np.ndarray,
    period: int = 20,
    mult: float = 1.0,
) -> tuple:
    """Efficiency Ratio bands — adaptive bands based on Kaufman's ER.

    ER = |close - close[period]| / sum(|close[i] - close[i-1]|)
    Bands widen in choppy markets (low ER) and tighten in trending (high ER).

    Returns (upper, mid, lower).
    """
    if closes.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()
    n = closes.size
    if n <= period:
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr.copy(), nan_arr.copy()

    # Efficiency Ratio
    direction = np.full(n, np.nan)
    direction[period:] = np.abs(closes[period:] - closes[:-period])

    volatility = np.full(n, np.nan)
    abs_diffs = np.abs(np.diff(closes))
    # Rolling sum of abs diffs over period
    cs_diffs = np.cumsum(abs_diffs)
    for i in range(period, n):
        volatility[i] = cs_diffs[i - 1] - (cs_diffs[i - period - 1] if i - period - 1 >= 0 else 0.0)

    er = np.full(n, np.nan)
    for i in range(period, n):
        if volatility[i] > 1e-12:
            er[i] = direction[i] / volatility[i]
        else:
            er[i] = 0.0

    # Adaptive EMA (KAMA-style midline)
    fast_sc = 2.0 / (2 + 1)
    slow_sc = 2.0 / (30 + 1)

    mid = np.full(n, np.nan)
    mid[period] = closes[period]
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        mid[i] = mid[i - 1] + sc * (closes[i] - mid[i - 1])

    # Band width based on rolling ATR-like measure (std of diffs)
    band_width = np.full(n, np.nan)
    for i in range(period, n):
        window = closes[i - period + 1 : i + 1]
        band_width[i] = np.std(window, ddof=1) * mult

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(mid[i]) and not np.isnan(band_width[i]):
            upper[i] = mid[i] + band_width[i]
            lower[i] = mid[i] - band_width[i]

    return upper, mid, lower
