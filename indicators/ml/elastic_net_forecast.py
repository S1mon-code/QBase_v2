import numpy as np
from sklearn.linear_model import ElasticNet


def elastic_net_signal(
    closes: np.ndarray,
    features_matrix: np.ndarray,
    period: int = 120,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Elastic Net regression signal.

    Combines L1 and L2 regularisation to predict 1-bar forward returns.
    Retrains every ``period // 4`` bars.

    Parameters
    ----------
    closes : (N,) price series.
    features_matrix : (N, K) feature array.
    period : training window length.
    alpha : overall regularisation strength.
    l1_ratio : balance between L1 (Lasso) and L2 (Ridge).

    Returns
    -------
    signal : (N,) predicted return direction.
    confidence : (N,) in-sample R² (higher = more confident).
    """
    n = len(closes)
    k = features_matrix.shape[1]
    signal = np.full(n, np.nan, dtype=np.float64)
    confidence = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return signal, confidence

    # 1-bar log return
    safe = np.maximum(closes, 1e-12)
    log_p = np.log(safe)
    ret1 = np.full(n, np.nan, dtype=np.float64)
    ret1[:-1] = log_p[1:] - log_p[:-1]

    retrain_every = max(1, period // 4)
    model: ElasticNet | None = None
    mean_x: np.ndarray | None = None
    std_x: np.ndarray | None = None
    r2: float = np.nan

    for i in range(period + 1, n):
        need_train = model is None or (i - period - 1) % retrain_every == 0

        if need_train:
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
            std_x = std_w
            X_normed = (X_t - mean_x) / std_x

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
            model.fit(X_normed, y_t)
            r2 = model.score(X_normed, y_t)

        if model is not None and mean_x is not None and std_x is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed = ((row - mean_x) / std_x).reshape(1, -1)
            signal[i] = model.predict(normed)[0]
            confidence[i] = r2

    return signal, confidence
