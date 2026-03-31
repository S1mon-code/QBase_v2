import numpy as np


def adaptive_lookback(closes: np.ndarray, min_period: int = 10,
                      max_period: int = 100) -> tuple:
    """Dynamically detect optimal lookback based on dominant cycle.

    Uses autocorrelation analysis to find the dominant cycle period
    in the price series and returns the optimal lookback.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    min_period : int
        Minimum allowed lookback period.
    max_period : int
        Maximum allowed lookback period.

    Returns
    -------
    optimal_period : np.ndarray
        Detected dominant cycle period per bar.
    cycle_strength : np.ndarray
        Strength of the dominant cycle (0-1). High = clear cycle.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    opt_period = np.full(n, np.nan)
    cyc_strength = np.full(n, np.nan)

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    window_size = max_period * 3

    for i in range(window_size, n):
        segment = rets[i - window_size + 1:i + 1]
        valid = segment[np.isfinite(segment)]
        if len(valid) < max_period * 2:
            continue

        mu = np.mean(valid)
        var = np.var(valid)
        if var < 1e-14:
            continue

        centered = valid - mu

        # Compute autocorrelation for lags from min_period to max_period
        best_lag = min_period
        best_acf = -1.0
        acf_values = []

        for lag in range(min_period, min(max_period + 1, len(centered) // 2)):
            acf = np.sum(centered[:len(centered) - lag] * centered[lag:]) / (
                len(centered) * var)
            acf_values.append(acf)

            if acf > best_acf:
                best_acf = acf
                best_lag = lag

        opt_period[i] = float(best_lag)

        # Cycle strength: how prominent the peak is
        if len(acf_values) >= 3:
            acf_arr = np.array(acf_values)
            mean_acf = np.mean(np.abs(acf_arr))
            if mean_acf > 1e-9:
                cyc_strength[i] = np.clip(best_acf / mean_acf / 3.0, 0.0, 1.0)
            else:
                cyc_strength[i] = 0.0

    return opt_period, cyc_strength
