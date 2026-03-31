"""Gold/Silver price ratio with rolling z-score.

Measures the relative valuation of gold vs silver.  A high z-score
suggests silver is undervalued relative to gold (historically cheap).
"""

import numpy as np


def gold_silver_ratio(
    au_closes: np.ndarray,
    ag_closes: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Gold/Silver price ratio + rolling z-score.

    Parameters
    ----------
    au_closes : array of gold closing prices.
    ag_closes : array of silver closing prices.
    period    : lookback window for z-score calculation.

    Returns
    -------
    (ratio, zscore)
        ratio   – AU / AG price ratio.
        zscore  – rolling z-score of the ratio.  High zscore means AG
                  is undervalued relative to AU.
    """
    n = len(au_closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    ratio = np.full(n, np.nan)
    zscore = np.full(n, np.nan)

    safe_ag = np.where(ag_closes == 0, np.nan, ag_closes)
    ratio = au_closes / safe_ag

    for i in range(period - 1, n):
        window = ratio[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            zscore[i] = (ratio[i] - mu) / sigma

    return ratio, zscore
