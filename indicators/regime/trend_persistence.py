import numpy as np


def trend_persistence(
    data: np.ndarray,
    max_lag: int = 20,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation-based trend persistence measure.

    Computes the sum of positive autocorrelations (lags 1..max_lag)
    within a rolling window. A high sum indicates persistent trending.
    The dominant lag is the lag with the highest autocorrelation.

    Returns (persistence, dominant_lag). High persistence = strong trend.
    """
    n = len(data)
    persistence = np.full(n, np.nan, dtype=np.float64)
    dominant_lag = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return persistence, dominant_lag

    safe = np.maximum(data, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    min_window = period + max_lag
    if len(log_ret) < min_window:
        return persistence, dominant_lag

    for i in range(min_window, len(log_ret) + 1):
        window = log_ret[i - period : i]
        mu = np.mean(window)
        var = np.var(window, ddof=0)

        if var < 1e-14:
            persistence[i] = 0.0
            dominant_lag[i] = 1.0
            continue

        centered = window - mu
        w_len = len(centered)

        best_ac = -np.inf
        best_lag = 1
        ac_sum = 0.0

        for lag in range(1, min(max_lag + 1, w_len)):
            ac = np.dot(centered[:w_len - lag], centered[lag:]) / (var * w_len)
            if ac > 0:
                ac_sum += ac
            if ac > best_ac:
                best_ac = ac
                best_lag = lag

        persistence[i] = ac_sum
        dominant_lag[i] = float(best_lag)

    return persistence, dominant_lag
