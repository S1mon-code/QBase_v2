import numpy as np


def oi_roc(
    oi: np.ndarray,
    period: int = 14,
) -> tuple:
    """Rate of Change of Open Interest.

    Measures OI momentum. Rising OI ROC = new money entering, falling = money leaving.
    Returns (oi_roc_val, oi_roc_signal).
    """
    n = len(oi)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    oi = oi.astype(np.float64)

    # OI ROC: (OI - OI[period ago]) / OI[period ago] * 100
    oi_roc_val = np.full(n, np.nan)
    for i in range(period, n):
        if oi[i - period] > 0:
            oi_roc_val[i] = (oi[i] - oi[i - period]) / oi[i - period] * 100.0

    # Signal line: EMA of OI ROC
    sig_period = max(period // 2, 3)
    oi_roc_signal = np.full(n, np.nan)
    alpha = 2.0 / (sig_period + 1)

    count = 0
    running_sum = 0.0
    seed_idx = -1
    for i in range(n):
        if np.isnan(oi_roc_val[i]):
            continue
        count += 1
        running_sum += oi_roc_val[i]
        if count == sig_period:
            oi_roc_signal[i] = running_sum / sig_period
            seed_idx = i
            break

    if seed_idx >= 0:
        for i in range(seed_idx + 1, n):
            if np.isnan(oi_roc_val[i]):
                oi_roc_signal[i] = oi_roc_signal[i - 1]
            else:
                oi_roc_signal[i] = oi_roc_signal[i - 1] * (1.0 - alpha) + oi_roc_val[i] * alpha

    return oi_roc_val, oi_roc_signal
