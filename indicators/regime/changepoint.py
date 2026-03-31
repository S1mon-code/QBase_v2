import numpy as np


def changepoint_score(
    data: np.ndarray,
    period: int = 60,
) -> np.ndarray:
    """Rolling changepoint probability based on mean/variance shifts.

    For each rolling window, tests every split point for the maximum
    log-likelihood ratio of a two-segment model (different mean/variance)
    vs a single-segment model. The ratio is normalized to [0, 1].

    Returns changepoint_probability (0-1 for each bar).
    """
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return out

    safe = np.maximum(data, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    if len(log_ret) < period:
        return out

    min_seg = max(5, period // 10)

    for i in range(period, len(log_ret) + 1):
        window = log_ret[i - period : i]
        w_len = len(window)

        total_var = np.var(window, ddof=1)
        if total_var < 1e-14:
            out[i] = 0.0
            continue

        # Single-segment log-likelihood (Gaussian)
        ll_single = -0.5 * w_len * (1.0 + np.log(2.0 * np.pi * total_var))

        best_ll_split = -np.inf

        for t in range(min_seg, w_len - min_seg + 1):
            seg1 = window[:t]
            seg2 = window[t:]

            var1 = np.var(seg1, ddof=1)
            var2 = np.var(seg2, ddof=1)

            if var1 < 1e-14 or var2 < 1e-14:
                continue

            ll1 = -0.5 * t * (1.0 + np.log(2.0 * np.pi * var1))
            ll2 = -0.5 * (w_len - t) * (1.0 + np.log(2.0 * np.pi * var2))
            ll_split = ll1 + ll2

            if ll_split > best_ll_split:
                best_ll_split = ll_split

        if best_ll_split == -np.inf:
            out[i] = 0.0
            continue

        # Log-likelihood ratio statistic
        lr = 2.0 * (best_ll_split - ll_single)
        lr = max(0.0, lr)

        # Normalize: use sigmoid-like mapping; LR ~ chi2(2) under null
        # P(chi2(2) > 9.21) = 0.01, so scale around that
        prob = 1.0 - np.exp(-lr / 6.0)
        out[i] = float(np.clip(prob, 0.0, 1.0))

    return out
