import numpy as np


def mean_crossing(closes: np.ndarray, period: int = 60) -> tuple:
    """Rate at which price crosses its rolling mean.

    Low crossing rate = trending (price stays on one side of mean).
    High crossing rate = ranging/mean-reverting.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    period : int
        Rolling window for mean and crossing rate calculation.

    Returns
    -------
    crossing_rate : np.ndarray
        Number of mean crossings per bar in the window (0-1 scale).
    crossing_rate_zscore : np.ndarray
        Z-score of crossing rate vs its own history.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    crossing_rate = np.full(n, np.nan)
    crossing_rate_z = np.full(n, np.nan)

    for i in range(period, n):
        window = closes[i - period:i + 1]
        if not np.all(np.isfinite(window)):
            valid = window[np.isfinite(window)]
            if len(valid) < period // 2:
                continue
            mu = np.mean(valid)
        else:
            mu = np.mean(window)

        # Count crossings
        above = window > mu
        crossings = 0
        for j in range(1, len(window)):
            if np.isfinite(window[j]) and np.isfinite(window[j - 1]):
                if above[j] != above[j - 1]:
                    crossings += 1

        crossing_rate[i] = crossings / (len(window) - 1)

    # Z-score of crossing rate
    lookback = period * 3
    for i in range(period + lookback, n):
        cr_window = crossing_rate[i - lookback:i + 1]
        valid = cr_window[np.isfinite(cr_window)]
        if len(valid) < lookback // 2:
            continue
        mu = np.mean(valid)
        std = np.std(valid)
        if std > 1e-9 and np.isfinite(crossing_rate[i]):
            crossing_rate_z[i] = (crossing_rate[i] - mu) / std

    return crossing_rate, crossing_rate_z
