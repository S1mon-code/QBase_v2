import numpy as np


def oi_concentration(oi: np.ndarray, period: int = 60) -> tuple:
    """OI percentile rank over rolling window.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    period : int
        Rolling window for percentile computation.

    Returns
    -------
    oi_percentile : np.ndarray
        OI percentile rank (0-100) within rolling window.
    oi_zscore : np.ndarray
        OI z-score within rolling window.
    is_extreme_high : np.ndarray (bool)
        True if OI is above 90th percentile.
    is_extreme_low : np.ndarray (bool)
        True if OI is below 10th percentile.
    """
    n = len(oi)
    if n == 0:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=bool), np.array([], dtype=bool))

    oi_pctl = np.full(n, np.nan)
    oi_z = np.full(n, np.nan)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)

    for i in range(period, n):
        window = oi[i - period:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < 10:
            continue

        cur = oi[i]
        if np.isnan(cur):
            continue

        # Percentile rank
        oi_pctl[i] = 100.0 * np.sum(valid <= cur) / len(valid)

        # Z-score
        mu = np.mean(valid)
        std = np.std(valid)
        if std > 0:
            oi_z[i] = (cur - mu) / std
        else:
            oi_z[i] = 0.0

        is_high[i] = oi_pctl[i] >= 90.0
        is_low[i] = oi_pctl[i] <= 10.0

    return oi_pctl, oi_z, is_high, is_low
