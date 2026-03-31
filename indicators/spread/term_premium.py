"""Term structure premium indicator for futures.

Quantifies contango vs backwardation and provides a z-scored measure
of the term premium.
"""

import numpy as np


def term_premium(
    front_closes: np.ndarray,
    back_closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Term structure premium (contango/backwardation).

    Parameters
    ----------
    front_closes : near-month contract closing prices.
    back_closes  : far-month contract closing prices.
    period       : lookback for z-score calculation.

    Returns
    -------
    (premium_pct, premium_zscore, is_contango)
        premium_pct    – (back - front) / front * 100.  Positive =
                         contango (back > front).
        premium_zscore – rolling z-score of premium_pct.
        is_contango    – boolean array; True when back > front.
    """
    n = len(front_closes)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, np.array([], dtype=bool)

    safe_front = np.where(front_closes == 0, np.nan, front_closes)
    premium_pct = (back_closes - front_closes) / safe_front * 100.0

    is_contango = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(premium_pct[i]):
            is_contango[i] = premium_pct[i] > 0

    premium_zscore = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = premium_pct[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            premium_zscore[i] = (premium_pct[i] - mu) / sigma

    return premium_pct, premium_zscore, is_contango
