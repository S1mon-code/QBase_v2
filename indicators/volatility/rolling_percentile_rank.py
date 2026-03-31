import numpy as np


def percentile_rank_features(
    features_matrix: np.ndarray,
    period: int = 60,
) -> np.ndarray:
    """Convert all features to rolling percentile ranks (0-100).

    Non-parametric normalization: for each feature, the value at bar i is
    replaced by its percentile rank within the trailing ``period`` bars.

    Parameters
    ----------
    features_matrix : (N, K) feature array.
    period : rolling window length.

    Returns
    -------
    ranked : (N, K) percentile-ranked features (0-100).  NaN during warmup.
    """
    n, k = features_matrix.shape
    ranked = np.full((n, k), np.nan, dtype=np.float64)

    if n < period:
        return ranked

    for j in range(k):
        col = features_matrix[:, j]
        for i in range(period, n):
            val = col[i]
            if np.isnan(val):
                continue
            window = col[i - period : i]
            valid = window[~np.isnan(window)]
            if len(valid) == 0:
                continue
            rank = np.sum(valid < val) / len(valid) * 100.0
            ranked[i, j] = rank

    return ranked
