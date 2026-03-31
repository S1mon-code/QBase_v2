"""3-asset relative value: is A rich or cheap vs B and C?

Computes a composite relative value score by comparing A's ratio
to both B and C, then z-scoring the result.
"""

import numpy as np


def rv_zscore(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    closes_c: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """3-asset relative value z-score.

    Parameters
    ----------
    closes_a : closing prices of target asset.
    closes_b : closing prices of reference asset B.
    closes_c : closing prices of reference asset C.
    period   : lookback window for z-score.

    Returns
    -------
    (rv_score, rv_zscore)
        rv_score  – composite relative value = mean(A/B z-score, A/C z-score).
                    Positive = A is rich, negative = A is cheap.
        rv_zscore – rolling z-score of the composite rv_score.
    """
    n = len(closes_a)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    safe_b = np.where(closes_b == 0, np.nan, closes_b)
    safe_c = np.where(closes_c == 0, np.nan, closes_c)

    ratio_ab = closes_a / safe_b
    ratio_ac = closes_a / safe_c

    def _rolling_zscore(arr: np.ndarray) -> np.ndarray:
        z = np.full(n, np.nan)
        for i in range(period - 1, n):
            window = arr[i - period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) < 2:
                continue
            mu = np.mean(valid)
            sigma = np.std(valid, ddof=1)
            if sigma > 0:
                z[i] = (arr[i] - mu) / sigma
        return z

    z_ab = _rolling_zscore(ratio_ab)
    z_ac = _rolling_zscore(ratio_ac)

    rv_score = np.full(n, np.nan)
    for i in range(n):
        vals = []
        if not np.isnan(z_ab[i]):
            vals.append(z_ab[i])
        if not np.isnan(z_ac[i]):
            vals.append(z_ac[i])
        if vals:
            rv_score[i] = np.mean(vals)

    # Z-score of the composite
    rv_z = _rolling_zscore(rv_score)

    return rv_score, rv_z
