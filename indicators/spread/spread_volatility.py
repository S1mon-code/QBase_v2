"""Volatility of the spread between two assets.

Spread vol expansion often precedes a breakout or regime change
in the relative value between two assets.
"""

import numpy as np


def spread_volatility(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling volatility of the price spread (A - B).

    Parameters
    ----------
    closes_a : closing prices of asset A.
    closes_b : closing prices of asset B.
    period   : lookback window for volatility calculation.

    Returns
    -------
    (spread_vol, spread_vol_zscore, is_volatile)
        spread_vol        – rolling standard deviation of the spread.
        spread_vol_zscore – z-score of spread_vol vs its own history.
        is_volatile       – 1.0 if zscore > 2, else 0.0.
    """
    n = len(closes_a)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()

    spread = closes_a - closes_b

    spread_vol = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = spread[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 2:
            spread_vol[i] = np.std(valid, ddof=1)

    # Z-score of spread_vol
    zperiod = max(period * 3, 60)
    sv_zscore = np.full(n, np.nan)
    is_vol = np.full(n, np.nan)

    for i in range(zperiod - 1, n):
        window = spread_vol[i - zperiod + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            sv_zscore[i] = (spread_vol[i] - mu) / sigma
            is_vol[i] = 1.0 if sv_zscore[i] > 2.0 else 0.0

    return spread_vol, sv_zscore, is_vol
