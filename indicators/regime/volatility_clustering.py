import numpy as np


def vol_clustering(
    data: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """GARCH-like volatility clustering measure.

    Measures autocorrelation of squared returns at lag 1 (clustering_score)
    and the sum of autocorrelations at lags 1-5 (persistence).
    High values indicate volatility begets volatility.

    Returns (clustering_score, persistence). High = vol begets vol.
    """
    n = len(data)
    clustering_score = np.full(n, np.nan, dtype=np.float64)
    persistence = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return clustering_score, persistence

    safe = np.maximum(data, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1
    sq_ret = log_ret ** 2

    if len(sq_ret) < period:
        return clustering_score, persistence

    max_ac_lag = 5

    for i in range(period, len(sq_ret) + 1):
        window = sq_ret[i - period : i]
        mu = np.mean(window)
        var = np.var(window, ddof=0)

        if var < 1e-14:
            clustering_score[i] = 0.0
            persistence[i] = 0.0
            continue

        centered = window - mu
        w_len = len(centered)

        # Lag-1 autocorrelation of squared returns
        ac1 = np.dot(centered[:w_len - 1], centered[1:]) / (var * w_len)
        clustering_score[i] = ac1

        # Sum of autocorrelations lags 1-5
        ac_sum = 0.0
        for lag in range(1, min(max_ac_lag + 1, w_len)):
            ac = np.dot(centered[:w_len - lag], centered[lag:]) / (var * w_len)
            ac_sum += ac
        persistence[i] = ac_sum

    return clustering_score, persistence
