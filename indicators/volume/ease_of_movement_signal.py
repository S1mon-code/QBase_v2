import numpy as np


def emv_signal(
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    period: int = 14,
    signal_period: int = 9,
) -> tuple:
    """Ease of Movement with signal line and histogram.

    EMV measures how easily price moves on volume.
    Returns (emv_line, emv_signal, emv_hist).
    """
    n = len(highs)
    if n < 2:
        emp = np.full(n, np.nan)
        return emp.copy(), emp.copy(), emp.copy()

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    volumes = volumes.astype(np.float64)

    # Distance moved: midpoint change
    mid = (highs + lows) / 2.0
    dm = np.empty(n)
    dm[0] = np.nan
    dm[1:] = mid[1:] - mid[:-1]

    # Box ratio: volume / (high - low)
    hl_range = highs - lows
    box_ratio = np.full(n, np.nan)
    mask = hl_range > 0
    box_ratio[mask] = volumes[mask] / hl_range[mask]

    # Raw EMV
    raw_emv = np.full(n, np.nan)
    valid = (~np.isnan(dm)) & (~np.isnan(box_ratio)) & (box_ratio > 0)
    raw_emv[valid] = dm[valid] / box_ratio[valid]

    # EMV line: SMA of raw EMV
    emv_line = np.full(n, np.nan)
    for i in range(period, n):
        window = raw_emv[i - period + 1 : i + 1]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) == period:
            emv_line[i] = np.mean(valid_w)

    # Signal line: SMA of EMV line
    emv_sig = np.full(n, np.nan)
    for i in range(n):
        start = i - signal_period + 1
        if start < 0:
            continue
        window = emv_line[start : i + 1]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) == signal_period:
            emv_sig[i] = np.mean(valid_w)

    # Histogram
    emv_hist = np.full(n, np.nan)
    both_valid = (~np.isnan(emv_line)) & (~np.isnan(emv_sig))
    emv_hist[both_valid] = emv_line[both_valid] - emv_sig[both_valid]

    return emv_line, emv_sig, emv_hist
