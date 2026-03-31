import numpy as np


def variance_ratio_test(
    data: np.ndarray,
    period: int = 60,
    holding: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Lo-MacKinlay variance ratio test for market efficiency.

    Compares the variance of `holding`-period returns to `holding` times
    the variance of 1-period returns. Under random walk, VR = 1.

    Returns (vr, vr_zscore). VR >> 1 = trending, VR << 1 = mean reverting.
    """
    n = len(data)
    vr = np.full(n, np.nan, dtype=np.float64)
    vr_zscore = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return vr, vr_zscore

    safe = np.maximum(data, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    if len(log_ret) < period:
        return vr, vr_zscore

    for i in range(period, len(log_ret) + 1):
        window = log_ret[i - period : i]
        w_len = len(window)

        if w_len < holding + 1:
            continue

        # 1-period variance
        var1 = np.var(window, ddof=1)
        if var1 < 1e-14:
            vr[i] = 1.0
            vr_zscore[i] = 0.0
            continue

        # holding-period returns
        n_hp = w_len - holding + 1
        hp_returns = np.array([
            np.sum(window[j : j + holding])
            for j in range(n_hp)
        ])
        var_hp = np.var(hp_returns, ddof=1)

        # Variance ratio
        ratio = var_hp / (holding * var1)
        vr[i] = ratio

        # Asymptotic z-score under the null of IID returns
        # Var(VR - 1) ~ 2(2q-1)(q-1) / (3q*T) for q=holding, T=w_len
        q = holding
        t = w_len
        avar = 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * t)
        if avar > 1e-14:
            vr_zscore[i] = (ratio - 1.0) / np.sqrt(avar)
        else:
            vr_zscore[i] = 0.0

    return vr, vr_zscore
