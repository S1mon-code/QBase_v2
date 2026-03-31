"""Cross-asset relative momentum.

Computes rate-of-change for two assets and returns their difference.
Positive values mean asset A is outperforming asset B.
"""

import numpy as np


def cross_momentum(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Relative momentum: ROC(A) - ROC(B).

    Parameters
    ----------
    closes_a : close prices of asset A.
    closes_b : close prices of asset B.
    period   : lookback for rate-of-change calculation.

    Returns
    -------
    np.ndarray
        Relative momentum.  Positive means A is outperforming B.
    """
    n = len(closes_a)
    if n == 0:
        return np.array([], dtype=float)
    if n <= period:
        return np.full(n, np.nan)

    rel_mom = np.full(n, np.nan)
    for i in range(period, n):
        prev_a = closes_a[i - period]
        prev_b = closes_b[i - period]
        if prev_a == 0 or prev_b == 0 or np.isnan(prev_a) or np.isnan(prev_b):
            continue
        roc_a = (closes_a[i] / prev_a - 1.0) * 100.0
        roc_b = (closes_b[i] / prev_b - 1.0) * 100.0
        rel_mom[i] = roc_a - roc_b

    return rel_mom
