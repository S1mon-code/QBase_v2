import numpy as np
from sklearn.linear_model import Lasso


def rolling_lasso_importance(
    closes: np.ndarray,
    features_matrix: np.ndarray,
    period: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Lasso feature selection: which features are predictive now?

    Fits Lasso on a trailing window, predicting 1-bar forward return.
    Absolute coefficient values indicate feature importance; zero
    coefficients mean the feature was dropped by L1 regularisation.

    Parameters
    ----------
    closes : (N,) price series.
    features_matrix : (N, K) feature array.
    period : training window length.

    Returns
    -------
    importance_matrix : (N, K) absolute coefficient values per feature.
    n_active_features : (N,) count of non-zero coefficients.
    """
    n = len(closes)
    k = features_matrix.shape[1]
    importance = np.full((n, k), np.nan, dtype=np.float64)
    n_active = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return importance, n_active

    # 1-bar log return as target
    safe = np.maximum(closes, 1e-12)
    log_p = np.log(safe)
    ret1 = np.full(n, np.nan, dtype=np.float64)
    ret1[:-1] = log_p[1:] - log_p[:-1]

    retrain_every = max(1, period // 4)
    coefs: np.ndarray | None = None

    for i in range(period + 1, n):
        need_train = coefs is None or (i - period - 1) % retrain_every == 0

        if need_train:
            # Use window ending 1 bar before current to avoid look-ahead
            start = i - period - 1
            end = i - 1
            X_train = features_matrix[start:end]
            y_train = ret1[start:end]

            valid = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
            if np.sum(valid) < max(10, k + 1):
                continue
            X_t = X_train[valid]
            y_t = y_train[valid]

            std_w = np.std(X_t, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean_x = np.mean(X_t, axis=0)
            X_t = (X_t - mean_x) / std_w

            model = Lasso(alpha=0.001, max_iter=2000)
            model.fit(X_t, y_t)
            coefs = np.abs(model.coef_)

        if coefs is not None:
            importance[i] = coefs
            n_active[i] = np.sum(coefs > 1e-10)

    return importance, n_active
