import numpy as np


def hurst_rs(data: np.ndarray, min_period: int = 10,
             max_period: int = 100) -> np.ndarray:
    """Classical R/S rescaled range Hurst exponent.

    Uses the rescaled range method (different from DFA-based Hurst in
    ``volatility/hurst.py``). H > 0.5 = persistent/trending,
    H < 0.5 = anti-persistent/mean-reverting.

    Parameters
    ----------
    data : np.ndarray
        Price series.
    min_period : int
        Minimum sub-period for R/S calculation.
    max_period : int
        Rolling window (also max sub-period).

    Returns
    -------
    hurst_value : np.ndarray
        Rolling Hurst exponent per bar.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=float)

    hurst_val = np.full(n, np.nan)

    safe = np.maximum(data, 1e-9)
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(safe[1:] / safe[:-1])

    for i in range(max_period, n):
        window = log_ret[i - max_period + 1:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < max_period // 2:
            continue

        # R/S analysis at multiple sub-period sizes
        log_ns = []
        log_rs = []

        sizes = []
        s = min_period
        while s <= len(valid) // 2:
            sizes.append(s)
            s = int(s * 1.5)
            if s == sizes[-1]:
                s += 1

        if len(sizes) < 3:
            continue

        for s in sizes:
            n_blocks = len(valid) // s
            if n_blocks < 1:
                continue

            rs_values = []
            for b in range(n_blocks):
                block = valid[b * s:(b + 1) * s]
                mu = np.mean(block)
                deviations = np.cumsum(block - mu)
                r = np.max(deviations) - np.min(deviations)
                std = np.std(block, ddof=1)
                if std > 1e-12:
                    rs_values.append(r / std)

            if len(rs_values) > 0:
                log_ns.append(np.log(s))
                log_rs.append(np.log(np.mean(rs_values)))

        if len(log_ns) < 3:
            continue

        # Linear regression: log(R/S) = H * log(n) + c
        log_ns = np.array(log_ns)
        log_rs = np.array(log_rs)
        x_mean = np.mean(log_ns)
        y_mean = np.mean(log_rs)
        ss_xx = np.sum((log_ns - x_mean) ** 2)
        if ss_xx > 1e-12:
            h = np.sum((log_ns - x_mean) * (log_rs - y_mean)) / ss_xx
            hurst_val[i] = np.clip(h, 0.0, 1.0)

    return hurst_val
