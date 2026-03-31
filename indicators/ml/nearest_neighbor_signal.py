import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def knn_signal(
    closes: np.ndarray,
    features_matrix: np.ndarray,
    period: int = 120,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling K-nearest-neighbor return prediction.

    Trains a KNN regressor on trailing ``period`` bars to predict forward
    returns from features.  Retrains every ``period // 4`` bars.

    Parameters
    ----------
    closes : (N,) price series.
    features_matrix : (N, K) feature array.
    period : training window length.
    k : number of neighbors.

    Returns
    -------
    prediction : (N,) predicted forward return.
    confidence : (N,) inverse mean distance to neighbors (higher = more confident).
    """
    n = len(closes)
    n_feat = features_matrix.shape[1]
    prediction = np.full(n, np.nan, dtype=np.float64)
    confidence = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return prediction, confidence

    # Compute 1-bar log returns as targets
    safe = np.maximum(closes, 1e-12)
    log_p = np.log(safe)
    returns = np.full(n, np.nan, dtype=np.float64)
    returns[1:] = log_p[1:] - log_p[:-1]

    retrain_every = max(1, period // 4)
    model: KNeighborsRegressor | None = None
    mean_x: np.ndarray | None = None
    std_x: np.ndarray | None = None

    for i in range(period + 1, n):
        need_train = model is None or (i - period - 1) % retrain_every == 0

        if need_train:
            # Features from [i-period-1, i-1), targets are returns at those bars
            start = i - period - 1
            end = i - 1
            X_train = features_matrix[start:end]
            y_train = returns[start + 1 : end + 1]

            valid = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
            if np.sum(valid) < max(k + 1, 10):
                continue
            X_t = X_train[valid]
            y_t = y_train[valid]

            std_w = np.std(X_t, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean_x = np.mean(X_t, axis=0)
            std_x = std_w
            X_t = (X_t - mean_x) / std_x

            actual_k = min(k, len(X_t) - 1)
            if actual_k < 1:
                continue
            model = KNeighborsRegressor(n_neighbors=actual_k)
            model.fit(X_t, y_t)

        if model is not None and mean_x is not None and std_x is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed = ((row - mean_x) / std_x).reshape(1, -1)
            prediction[i] = model.predict(normed)[0]
            dists, _ = model.kneighbors(normed)
            mean_dist = np.mean(dists)
            confidence[i] = 1.0 / (1.0 + mean_dist)

    return prediction, confidence
