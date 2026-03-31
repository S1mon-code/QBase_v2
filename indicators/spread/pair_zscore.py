"""Pair trading z-score with half-life estimation.

Supports ratio-based and difference-based spread construction.
"""

import numpy as np


def pair_zscore(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    period: int = 60,
    method: str = "ratio",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair trading z-score.

    Parameters
    ----------
    closes_a : close prices of asset A.
    closes_b : close prices of asset B.
    period   : lookback window for mean/std and half-life.
    method   : 'ratio' (A/B) or 'diff' (A - B).

    Returns
    -------
    (spread, zscore, half_life_estimate)
        spread             – raw spread series.
        zscore             – rolling z-score of the spread.
        half_life_estimate – rolling estimate of mean-reversion
                             half-life (in bars) via AR(1) regression.
    """
    n = len(closes_a)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    # Build spread
    if method == "ratio":
        safe_b = np.where(closes_b == 0, np.nan, closes_b)
        spread = closes_a / safe_b
    else:
        spread = closes_a - closes_b

    zscore = np.full(n, np.nan)
    half_life = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = spread[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue

        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            zscore[i] = (spread[i] - mu) / sigma

        # Half-life via AR(1): spread_t = phi * spread_{t-1} + eps
        y = valid[1:] - mu
        x = valid[:-1] - mu
        denom = np.dot(x, x)
        if denom > 0:
            phi = np.dot(x, y) / denom
            if 0 < phi < 1:
                half_life[i] = -np.log(2) / np.log(phi)

    return spread, zscore, half_life
