import numpy as np
from sklearn.ensemble import RandomForestClassifier


def rolling_tree_importance(
    closes: np.ndarray,
    features_matrix: np.ndarray,
    period: int = 120,
    n_estimators: int = 50,
) -> np.ndarray:
    """Rolling Random Forest feature importance for predicting return direction.

    Trains a classifier on a trailing window to predict whether the next
    bar's return is positive or negative.  Feature importances (Gini-based)
    indicate which features are most predictive in the current regime.
    Retrains every ``period // 4`` bars.

    Parameters
    ----------
    closes : (N,) price series.
    features_matrix : (N, K) feature array.
    period : training window length.
    n_estimators : number of trees in the forest.

    Returns
    -------
    importance_matrix : (N, K) feature importance values per bar.
    """
    n = len(closes)
    k = features_matrix.shape[1]
    importance = np.full((n, k), np.nan, dtype=np.float64)

    if n < period + 1:
        return importance

    # 1-bar return direction as target
    safe = np.maximum(closes, 1e-12)
    log_p = np.log(safe)
    direction = np.full(n, np.nan, dtype=np.float64)
    direction[:-1] = np.sign(log_p[1:] - log_p[:-1])

    retrain_every = max(1, period // 4)
    importances: np.ndarray | None = None

    for i in range(period + 1, n):
        need_train = importances is None or (i - period - 1) % retrain_every == 0

        if need_train:
            start = i - period - 1
            end = i - 1
            X_train = features_matrix[start:end]
            y_train = direction[start:end]

            valid = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
            if np.sum(valid) < max(20, k + 1):
                continue
            X_t = X_train[valid]
            y_t = y_train[valid]

            # Need at least 2 classes
            unique = np.unique(y_t)
            if len(unique) < 2:
                continue

            std_w = np.std(X_t, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean_x = np.mean(X_t, axis=0)
            X_t = (X_t - mean_x) / std_w

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                random_state=42,
            )
            model.fit(X_t, y_t)
            importances = model.feature_importances_

        if importances is not None:
            importance[i] = importances

    return importance
