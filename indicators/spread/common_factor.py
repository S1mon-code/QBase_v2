"""Extract common factor from multiple assets (first principal component).

Useful for identifying the dominant driver across a basket of
correlated assets and measuring individual loadings.
"""

import numpy as np


def common_factor(
    closes_list: list[np.ndarray],
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling first principal component across multiple assets.

    Parameters
    ----------
    closes_list : list of closing price arrays (all same length).
    period      : rolling window for PCA computation.

    Returns
    -------
    (common_factor, loadings)
        common_factor – first PC score at each bar (same length as inputs).
        loadings      – latest factor loadings, shape (n_assets,).
                        NaN-filled until warmup completes.
    """
    if len(closes_list) == 0 or len(closes_list[0]) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_assets = len(closes_list)
    n = len(closes_list[0])

    # Compute returns for each asset
    returns = np.full((n, n_assets), np.nan)
    for j in range(n_assets):
        for i in range(1, n):
            if closes_list[j][i - 1] != 0 and not np.isnan(closes_list[j][i - 1]):
                returns[i, j] = closes_list[j][i] / closes_list[j][i - 1] - 1.0

    factor = np.full(n, np.nan)
    loadings = np.full(n_assets, np.nan)

    for i in range(period - 1, n):
        window = returns[i - period + 1 : i + 1, :]
        # Check for sufficient valid data
        valid_mask = ~np.any(np.isnan(window), axis=1)
        valid_rows = window[valid_mask]
        if len(valid_rows) < max(n_assets + 1, 10):
            continue

        # Demean
        means = np.mean(valid_rows, axis=0)
        centred = valid_rows - means

        # Covariance and eigen decomposition
        cov = centred.T @ centred / (len(valid_rows) - 1)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue

        # First PC = eigenvector with largest eigenvalue (last one from eigh)
        pc1 = eigenvectors[:, -1]

        # Project current returns onto PC1
        cur_ret = returns[i, :]
        if not np.any(np.isnan(cur_ret)):
            factor[i] = np.dot(cur_ret - means, pc1)
            loadings = pc1.copy()

    return factor, loadings
