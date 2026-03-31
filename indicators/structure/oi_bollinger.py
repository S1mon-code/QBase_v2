import numpy as np


def oi_bollinger(oi: np.ndarray, period: int = 20,
                 num_std: float = 2.0) -> tuple:
    """Bollinger Bands on Open Interest.

    OI at the upper band indicates extremely crowded positioning;
    OI at the lower band indicates thin positioning.

    Parameters
    ----------
    oi : np.ndarray
        Open interest series.
    period : int
        Lookback window for mean and std.
    num_std : float
        Number of standard deviations for bands.

    Returns
    -------
    upper : np.ndarray
        Upper Bollinger Band on OI.
    mid : np.ndarray
        Middle band (SMA of OI).
    lower : np.ndarray
        Lower Bollinger Band on OI.
    oi_zscore : np.ndarray
        Z-score of OI relative to its rolling mean/std.
    """
    n = len(oi)
    upper = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    oi_zscore = np.full(n, np.nan)

    if n < period:
        return upper, mid, lower, oi_zscore

    for i in range(period - 1, n):
        window = oi[i - period + 1:i + 1]
        m = np.mean(window)
        s = np.std(window, ddof=1)
        mid[i] = m
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s
        if s > 0:
            oi_zscore[i] = (oi[i] - m) / s
        else:
            oi_zscore[i] = 0.0

    return upper, mid, lower, oi_zscore
