"""Up-volume vs down-volume imbalance.

Classifies each bar's volume as buying or selling based on
price direction, then computes a rolling imbalance ratio.
"""

import numpy as np


def volume_imbalance(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Volume imbalance: net buying vs selling pressure.

    Parameters
    ----------
    closes  : array of closing prices.
    volumes : array of trading volumes.
    period  : rolling window.

    Returns
    -------
    (imbalance, imbalance_signal)
        imbalance        – (up_vol - down_vol) / total_vol, range [-1, 1].
                           +1 = all buying, -1 = all selling.
        imbalance_signal – smoothed imbalance (EMA).
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Classify volume
    up_vol = np.full(n, np.nan)
    down_vol = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(closes[i]) or np.isnan(closes[i - 1]) or np.isnan(volumes[i]):
            continue
        if closes[i] > closes[i - 1]:
            up_vol[i] = volumes[i]
            down_vol[i] = 0.0
        elif closes[i] < closes[i - 1]:
            up_vol[i] = 0.0
            down_vol[i] = volumes[i]
        else:
            up_vol[i] = volumes[i] * 0.5
            down_vol[i] = volumes[i] * 0.5

    imbalance = np.full(n, np.nan)
    for i in range(period, n):
        uv = up_vol[i - period + 1 : i + 1]
        dv = down_vol[i - period + 1 : i + 1]
        mask = ~(np.isnan(uv) | np.isnan(dv))
        if np.sum(mask) < 5:
            continue
        total_up = np.sum(uv[mask])
        total_down = np.sum(dv[mask])
        total = total_up + total_down
        if total > 0:
            imbalance[i] = (total_up - total_down) / total

    # EMA smoothing
    alpha = 2.0 / (period + 1)
    signal = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(imbalance[i]):
            continue
        if np.isnan(signal[i - 1]) if i > 0 else True:
            signal[i] = imbalance[i]
        else:
            signal[i] = alpha * imbalance[i] + (1 - alpha) * signal[i - 1]

    return imbalance, signal
