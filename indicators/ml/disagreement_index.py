import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


def model_disagreement(
    closes: np.ndarray,
    features_matrix: np.ndarray,
    period: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Train 3 diverse models and measure prediction disagreement.

    Uses Ridge, Random Forest, and KNN regressors.  High disagreement
    signals an uncertain regime.  Retrains every ``period // 4`` bars.

    Parameters
    ----------
    closes : (N,) price series.
    features_matrix : (N, K) feature array.
    period : training window length.

    Returns
    -------
    disagreement : (N,) standard deviation of predictions across 3 models.
    avg_prediction : (N,) mean prediction across 3 models.
    """
    n = len(closes)
    k = features_matrix.shape[1]
    disagreement = np.full(n, np.nan, dtype=np.float64)
    avg_prediction = np.full(n, np.nan, dtype=np.float64)

    if n < period + 2:
        return disagreement, avg_prediction

    # 1-bar forward returns as target
    safe = np.maximum(closes, 1e-12)
    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    fwd_ret[:-1] = safe[1:] / safe[:-1] - 1.0

    retrain_every = max(1, period // 4)
    models = [None, None, None]
    mean_x: np.ndarray | None = None
    std_x: np.ndarray | None = None

    for i in range(period + 1, n):
        need_train = models[0] is None or (i - period - 1) % retrain_every == 0

        if need_train:
            start = i - period - 1
            end = i - 1
            X_train = features_matrix[start:end]
            y_train = fwd_ret[start:end]

            valid = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
            if np.sum(valid) < max(10, k + 1):
                continue
            X_t = X_train[valid]
            y_t = y_train[valid]

            std_w = np.std(X_t, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean_x = np.mean(X_t, axis=0)
            std_x = std_w
            X_t = (X_t - mean_x) / std_x

            m1 = Ridge(alpha=1.0)
            m1.fit(X_t, y_t)

            m2 = RandomForestRegressor(
                n_estimators=10, max_depth=4, random_state=42
            )
            m2.fit(X_t, y_t)

            n_neighbors = min(5, len(X_t) - 1)
            if n_neighbors < 1:
                continue
            m3 = KNeighborsRegressor(n_neighbors=n_neighbors)
            m3.fit(X_t, y_t)

            models = [m1, m2, m3]

        if models[0] is not None and mean_x is not None and std_x is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed = ((row - mean_x) / std_x).reshape(1, -1)

            preds = np.array([m.predict(normed)[0] for m in models])
            disagreement[i] = np.std(preds)
            avg_prediction[i] = np.mean(preds)

    return disagreement, avg_prediction
