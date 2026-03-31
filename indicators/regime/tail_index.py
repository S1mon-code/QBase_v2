import numpy as np


def hill_tail_index(
    data: np.ndarray,
    period: int = 60,
    k_fraction: float = 0.1,
) -> np.ndarray:
    """Hill estimator for tail index (fat-tailedness).

    Estimates the tail index alpha from the largest absolute returns
    in each rolling window. Uses the top `k_fraction` of observations.

    Returns tail_index. < 2 = extremely fat tails, > 4 = near-normal.
    """
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return out

    safe = np.maximum(data, 1e-9)
    log_ret = np.abs(np.diff(np.log(safe)))  # absolute log returns, length n-1

    if len(log_ret) < period:
        return out

    k = max(2, int(period * k_fraction))

    for i in range(period, len(log_ret) + 1):
        window = log_ret[i - period : i]

        # Sort descending, take top k
        sorted_vals = np.sort(window)[::-1]
        top_k = sorted_vals[:k]

        # Threshold is the (k+1)-th largest value
        if k >= len(sorted_vals):
            threshold = sorted_vals[-1]
        else:
            threshold = sorted_vals[k]

        if threshold < 1e-14:
            # All values near zero; can't estimate tail
            out[i] = np.nan
            continue

        # Hill estimator: 1 / (mean of log(X_i / X_{k+1}))
        log_ratios = np.log(top_k / threshold)
        mean_log = np.mean(log_ratios)

        if mean_log < 1e-14:
            out[i] = np.nan
            continue

        alpha = 1.0 / mean_log
        out[i] = float(np.clip(alpha, 0.5, 10.0))

    return out
