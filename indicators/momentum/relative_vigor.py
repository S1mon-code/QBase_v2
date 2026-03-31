import numpy as np


def relative_vigor_index(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 10,
) -> tuple:
    """Relative Vigor Index (close-open numerator, high-low denominator).

    Uses a 4-bar symmetric weighted moving average for smoothing,
    then sums over `period` bars.

    Returns (rvi, signal) where signal is a 4-bar symmetric WMA of rvi.
    """
    if closes.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy()
    n = closes.size
    if n < period + 6:
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr.copy()

    # Symmetric FIR weights: [1, 2, 2, 1] / 6
    co = closes - opens  # close - open
    hl = highs - lows    # high - low

    # 4-bar symmetric weighted average
    co_smooth = np.full(n, np.nan)
    hl_smooth = np.full(n, np.nan)
    for i in range(3, n):
        co_smooth[i] = (co[i] + 2.0 * co[i - 1] + 2.0 * co[i - 2] + co[i - 3]) / 6.0
        hl_smooth[i] = (hl[i] + 2.0 * hl[i - 1] + 2.0 * hl[i - 2] + hl[i - 3]) / 6.0

    # Sum over period
    rvi = np.full(n, np.nan)
    for i in range(3 + period - 1, n):
        num = np.nansum(co_smooth[i - period + 1 : i + 1])
        den = np.nansum(hl_smooth[i - period + 1 : i + 1])
        if abs(den) > 1e-12:
            rvi[i] = num / den
        else:
            rvi[i] = 0.0

    # Signal line: 4-bar symmetric WMA of RVI
    signal = np.full(n, np.nan)
    for i in range(3 + period - 1 + 3, n):
        signal[i] = (rvi[i] + 2.0 * rvi[i - 1] + 2.0 * rvi[i - 2] + rvi[i - 3]) / 6.0

    return rvi, signal
