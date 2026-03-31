import numpy as np


def volume_force(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 13,
) -> tuple:
    """Volume Force indicator.

    Combines price change direction and magnitude with volume to gauge
    buying/selling pressure. Positive = buying dominance, negative = selling.
    Returns (vf, vf_signal).
    """
    n = len(closes)
    if n < 2:
        emp = np.full(n, np.nan)
        return emp.copy(), emp.copy()

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    # Raw volume force: price_change_pct * volume
    raw_vf = np.empty(n)
    raw_vf[0] = np.nan
    for i in range(1, n):
        if closes[i - 1] > 0:
            pct_change = (closes[i] - closes[i - 1]) / closes[i - 1]
            raw_vf[i] = pct_change * volumes[i]
        else:
            raw_vf[i] = 0.0

    # EMA smoothing
    vf = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)

    # Seed
    count = 0
    running_sum = 0.0
    seed_idx = -1
    for i in range(1, n):
        if np.isnan(raw_vf[i]):
            continue
        count += 1
        running_sum += raw_vf[i]
        if count == period:
            vf[i] = running_sum / period
            seed_idx = i
            break

    if seed_idx >= 0:
        for i in range(seed_idx + 1, n):
            if np.isnan(raw_vf[i]):
                vf[i] = vf[i - 1]
            else:
                vf[i] = vf[i - 1] * (1.0 - alpha) + raw_vf[i] * alpha

    # Signal line: EMA of VF
    sig_period = max(period // 2, 3)
    vf_signal = np.full(n, np.nan)
    alpha_s = 2.0 / (sig_period + 1)

    count = 0
    running_sum = 0.0
    seed_idx = -1
    for i in range(n):
        if np.isnan(vf[i]):
            continue
        count += 1
        running_sum += vf[i]
        if count == sig_period:
            vf_signal[i] = running_sum / sig_period
            seed_idx = i
            break

    if seed_idx >= 0:
        for i in range(seed_idx + 1, n):
            if np.isnan(vf[i]):
                vf_signal[i] = vf_signal[i - 1]
            else:
                vf_signal[i] = vf_signal[i - 1] * (1.0 - alpha_s) + vf[i] * alpha_s

    return vf, vf_signal
