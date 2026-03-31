import numpy as np


def vol_of_vol_regime(
    closes: np.ndarray,
    vol_period: int = 20,
    vov_period: int = 20,
) -> tuple:
    """Vol-of-vol with 3-state regime classification.

    Computes rolling volatility, then the volatility of that volatility.
    Classifies into: stable (low VoV), transitioning (medium VoV), crisis (high VoV).
    Returns (vov, is_stable, is_transitioning, is_crisis).
    Boolean arrays are 1.0/0.0/NaN.
    """
    n = len(closes)
    if n < 2:
        emp = np.full(n, np.nan)
        return emp.copy(), emp.copy(), emp.copy(), emp.copy()

    closes = closes.astype(np.float64)
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(closes[1:] / closes[:-1])

    # Rolling volatility
    vol = np.full(n, np.nan)
    for i in range(vol_period, n):
        window = log_ret[i - vol_period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 5:
            vol[i] = np.std(valid, ddof=1)

    # Vol-of-vol: std of vol over vov_period
    vov = np.full(n, np.nan)
    total_warmup = vol_period + vov_period
    for i in range(total_warmup, n):
        window = vol[i - vov_period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 5:
            mu = np.mean(valid)
            if mu > 0:
                vov[i] = np.std(valid, ddof=1) / mu  # coefficient of variation

    # Regime classification using expanding percentiles
    is_stable = np.full(n, np.nan)
    is_transitioning = np.full(n, np.nan)
    is_crisis = np.full(n, np.nan)

    lookback = max(60, vov_period * 3)
    for i in range(total_warmup, n):
        start = max(0, i - lookback + 1)
        history = vov[start : i + 1]
        valid = history[~np.isnan(history)]
        if len(valid) < 10:
            continue

        p33 = np.percentile(valid, 33)
        p67 = np.percentile(valid, 67)
        v = vov[i]
        if np.isnan(v):
            continue

        if v <= p33:
            is_stable[i] = 1.0
            is_transitioning[i] = 0.0
            is_crisis[i] = 0.0
        elif v <= p67:
            is_stable[i] = 0.0
            is_transitioning[i] = 1.0
            is_crisis[i] = 0.0
        else:
            is_stable[i] = 0.0
            is_transitioning[i] = 0.0
            is_crisis[i] = 1.0

    return vov, is_stable, is_transitioning, is_crisis
