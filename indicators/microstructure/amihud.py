"""Amihud (2002) illiquidity ratio.

Measures price impact per unit of volume.  Higher values indicate
less liquid markets where trades move prices more.
"""

import numpy as np


def amihud_illiquidity(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Amihud illiquidity ratio: |return| / volume.

    Parameters
    ----------
    closes  : closing prices.
    volumes : trading volumes.
    period  : rolling average window.

    Returns
    -------
    (illiquidity, illiquidity_zscore)
        illiquidity       – rolling average of daily |return| / volume.
        illiquidity_zscore – rolling z-score of the illiquidity ratio.
                            High values signal deteriorating liquidity.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan)

    # Daily |return| / volume
    raw = np.full(n, np.nan)
    for i in range(1, n):
        if closes[i - 1] != 0 and not np.isnan(closes[i - 1]):
            abs_ret = abs(closes[i] / closes[i - 1] - 1.0)
            vol = volumes[i]
            if vol > 0 and not np.isnan(vol):
                raw[i] = abs_ret / vol

    illiq = np.full(n, np.nan)
    for i in range(period, n):
        window = raw[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            illiq[i] = np.mean(valid)

    # Z-score
    zscore = np.full(n, np.nan)
    zperiod = max(period * 3, 60)
    for i in range(zperiod - 1, n):
        window = illiq[i - zperiod + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            zscore[i] = (illiq[i] - mu) / sigma

    return illiq, zscore
