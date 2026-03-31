"""Generic metal ratio indicator with z-score and percentile rank.

Works for any pair of metals or commodities: CU/AG, AU/AG, etc.
"""

import numpy as np


def metal_ratio(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generic metal ratio (A/B) + z-score + percentile rank.

    Parameters
    ----------
    closes_a : close prices of asset A (numerator).
    closes_b : close prices of asset B (denominator).
    period   : lookback window for z-score and percentile.

    Returns
    -------
    (ratio, zscore, percentile_rank)
        ratio           – A / B price ratio.
        zscore          – rolling z-score of the ratio.
        percentile_rank – rolling percentile rank (0-100) of current
                          ratio within the lookback window.
    """
    n = len(closes_a)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    safe_b = np.where(closes_b == 0, np.nan, closes_b)
    ratio = closes_a / safe_b

    zscore = np.full(n, np.nan)
    pct_rank = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = ratio[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            zscore[i] = (ratio[i] - mu) / sigma
        # percentile rank: fraction of values <= current
        pct_rank[i] = np.sum(valid <= ratio[i]) / len(valid) * 100.0

    return ratio, zscore, pct_rank
