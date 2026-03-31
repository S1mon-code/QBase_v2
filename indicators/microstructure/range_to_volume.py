"""Range per unit volume — price sensitivity to volume.

High range-to-volume indicates thin markets where small volume
causes large price swings.  Low values suggest deep liquidity.
"""

import numpy as np


def range_to_volume(
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Range-to-volume ratio.

    Parameters
    ----------
    highs   : array of high prices.
    lows    : array of low prices.
    volumes : array of trading volumes.
    period  : rolling window.

    Returns
    -------
    (rtv, rtv_zscore)
        rtv        – rolling average of (high - low) / volume.
        rtv_zscore – rolling z-score of rtv.
    """
    n = len(highs)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    raw = np.full(n, np.nan)
    for i in range(n):
        if (not np.isnan(highs[i]) and not np.isnan(lows[i])
                and not np.isnan(volumes[i]) and volumes[i] > 0):
            raw[i] = (highs[i] - lows[i]) / volumes[i]

    rtv = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = raw[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rtv[i] = np.mean(valid)

    # Z-score
    zperiod = max(period * 3, 60)
    zscore = np.full(n, np.nan)
    for i in range(zperiod - 1, n):
        window = rtv[i - zperiod + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            zscore[i] = (rtv[i] - mu) / sigma

    return rtv, zscore
