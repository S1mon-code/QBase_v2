import numpy as np
from sklearn.covariance import LedoitWolf


def ledoit_wolf_features(returns_matrix, period=120):
    """Rolling Ledoit-Wolf shrinkage covariance features.

    Fits a Ledoit-Wolf estimator on each rolling window and extracts
    summary statistics.  The model is retrained every ``period // 4``
    bars to save computation.

    Parameters
    ----------
    returns_matrix : (N, K) array of return series (K >= 2).
    period : rolling window size.

    Returns
    -------
    shrinkage_intensity : (N,) LW shrinkage coefficient.
        High = noisy / unstable correlation structure.
    implied_correlation : (N,) average off-diagonal correlation from
        the shrunk covariance.
    max_eigenvalue : (N,) largest eigenvalue of the shrunk covariance.
    """
    returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
    n, k = returns_matrix.shape
    shrinkage_intensity = np.full(n, np.nan, dtype=np.float64)
    implied_correlation = np.full(n, np.nan, dtype=np.float64)
    max_eigenvalue = np.full(n, np.nan, dtype=np.float64)

    if n < period or k < 2:
        return shrinkage_intensity, implied_correlation, max_eigenvalue

    retrain_interval = max(1, period // 4)
    last_shrinkage = np.nan
    last_corr = np.nan
    last_eig = np.nan

    for i in range(period - 1, n):
        need_retrain = (i == period - 1) or ((i - (period - 1)) % retrain_interval == 0)

        if need_retrain:
            win = returns_matrix[i - period + 1: i + 1]
            # drop NaN rows
            mask = ~np.any(np.isnan(win), axis=1)
            win_clean = win[mask]
            if len(win_clean) < k + 2:
                continue

            try:
                lw = LedoitWolf().fit(win_clean)
            except Exception:
                continue

            cov = lw.covariance_
            last_shrinkage = lw.shrinkage_

            # implied correlation
            stds = np.sqrt(np.diag(cov))
            stds[stds < 1e-10] = 1e-10
            corr_matrix = cov / np.outer(stds, stds)
            # average off-diagonal
            mask_tri = np.triu(np.ones((k, k), dtype=bool), k=1)
            last_corr = corr_matrix[mask_tri].mean()

            # max eigenvalue
            eigvals = np.linalg.eigvalsh(cov)
            last_eig = eigvals.max()

        shrinkage_intensity[i] = last_shrinkage
        implied_correlation[i] = last_corr
        max_eigenvalue[i] = last_eig

    return shrinkage_intensity, implied_correlation, max_eigenvalue
