"""Rolling trade intensity — volume acceleration.

Measures how quickly volume is changing, capturing surges
in participation that often precede large price moves.
"""

import numpy as np


def trade_intensity(
    volumes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Trade intensity = volume acceleration.

    Parameters
    ----------
    volumes : array of trading volumes.
    period  : lookback window.

    Returns
    -------
    (intensity, intensity_zscore)
        intensity        – ratio of current volume to rolling mean volume.
        intensity_zscore – rolling z-score of intensity.
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    intensity = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = volumes[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            avg = np.mean(valid)
            if avg > 0:
                intensity[i] = volumes[i] / avg

    # Z-score of intensity
    zperiod = max(period * 3, 60)
    zscore = np.full(n, np.nan)
    for i in range(zperiod - 1, n):
        window = intensity[i - zperiod + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            zscore[i] = (intensity[i] - mu) / sigma

    return intensity, zscore
