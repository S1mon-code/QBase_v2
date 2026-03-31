import numpy as np


def turbulence(
    returns_matrix: np.ndarray,
    period: int = 60,
) -> np.ndarray:
    """Mahalanobis distance from historical mean -- market turbulence.

    For a single asset, pass 1D price array; lagged returns are used as features.
    For multi-asset, pass (N, K) array of K asset returns.

    Returns turbulence_score. High = unusual market conditions.
    """
    if returns_matrix.ndim == 1:
        return _turbulence_single(returns_matrix, period)
    return _turbulence_multi(returns_matrix, period)


def _turbulence_single(
    data: np.ndarray,
    period: int,
) -> np.ndarray:
    """Single-asset turbulence using lagged returns as features."""
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return out

    safe = np.maximum(data, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    n_lags = 5
    min_len = period + n_lags

    if len(log_ret) < min_len:
        return out

    # Build lagged feature matrix
    feat_len = len(log_ret) - n_lags + 1
    features = np.column_stack([
        log_ret[n_lags - 1 - lag : len(log_ret) - lag]
        for lag in range(n_lags)
    ])  # shape (feat_len, n_lags)

    # Map feature index back to original index: feature row j -> original index j + n_lags
    for i in range(period, feat_len):
        window = features[i - period : i]
        current = features[i]

        mu = np.mean(window, axis=0)
        cov = np.cov(window, rowvar=False)

        if cov.ndim < 2:
            continue

        # Regularize covariance
        cov += np.eye(n_lags) * 1e-8

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue

        diff = current - mu
        turb = float(diff @ cov_inv @ diff)
        # Map back: feature index i -> original data index i + n_lags
        orig_idx = i + n_lags
        if orig_idx < n:
            out[orig_idx] = turb

    return out


def _turbulence_multi(
    returns_matrix: np.ndarray,
    period: int,
) -> np.ndarray:
    """Multi-asset turbulence using Mahalanobis distance."""
    n, k = returns_matrix.shape
    out = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1 or k < 1:
        return out

    for i in range(period, n):
        window = returns_matrix[i - period : i]
        current = returns_matrix[i]

        mu = np.mean(window, axis=0)
        cov = np.cov(window, rowvar=False)

        if cov.ndim < 2:
            cov = np.atleast_2d(cov)

        cov += np.eye(k) * 1e-8

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue

        diff = current - mu
        out[i] = float(diff @ cov_inv @ diff)

    return out
