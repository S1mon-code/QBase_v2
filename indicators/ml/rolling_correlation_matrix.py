import numpy as np


def rolling_eigen_features(features_matrix, period=60):
    """Rolling eigenvalue decomposition of feature correlation matrix.

    For each rolling window, computes the correlation matrix of features,
    extracts eigenvalues, and returns summary statistics that capture
    systemic risk / market fragility.

    Parameters
    ----------
    features_matrix : (N, K) array of features (K >= 2).
    period : rolling window size.

    Returns
    -------
    max_eigenvalue : (N,) largest eigenvalue of the correlation matrix.
    eigenvalue_ratio : (N,) ratio of largest to smallest eigenvalue (condition).
    absorption_ratio : (N,) fraction of variance explained by top eigenvalue(s).
        High absorption_ratio = features highly correlated = fragile market.
    """
    features_matrix = np.asarray(features_matrix, dtype=np.float64)
    n, k = features_matrix.shape
    max_eigenvalue = np.full(n, np.nan, dtype=np.float64)
    eigenvalue_ratio = np.full(n, np.nan, dtype=np.float64)
    absorption_ratio = np.full(n, np.nan, dtype=np.float64)

    if n < period or k < 2:
        return max_eigenvalue, eigenvalue_ratio, absorption_ratio

    for i in range(period - 1, n):
        win = features_matrix[i - period + 1: i + 1]
        if np.any(np.isnan(win)):
            mask = ~np.any(np.isnan(win), axis=1)
            if mask.sum() < k + 1:
                continue
            win = win[mask]

        # standardise columns
        std = win.std(axis=0)
        if np.any(std < 1e-10):
            continue
        win_norm = (win - win.mean(axis=0)) / std

        # correlation matrix
        corr = (win_norm.T @ win_norm) / (len(win_norm) - 1)

        # eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(corr)
        except np.linalg.LinAlgError:
            continue

        eigvals = np.sort(eigvals)[::-1]  # descending
        eigvals = np.maximum(eigvals, 0)  # clip numerical negatives

        total_var = eigvals.sum()
        if total_var < 1e-10:
            continue

        max_eigenvalue[i] = eigvals[0]

        min_eig = eigvals[-1]
        eigenvalue_ratio[i] = eigvals[0] / max(min_eig, 1e-10)

        # absorption ratio: top ceil(k/5) eigenvalues explain how much?
        n_top = max(1, int(np.ceil(k / 5)))
        absorption_ratio[i] = eigvals[:n_top].sum() / total_var

    return max_eigenvalue, eigenvalue_ratio, absorption_ratio
