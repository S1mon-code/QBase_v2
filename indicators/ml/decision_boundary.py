import numpy as np
from sklearn.svm import SVC


def decision_boundary_distance(
    features_matrix: np.ndarray,
    labels: np.ndarray,
    period: int = 120,
) -> np.ndarray:
    """Rolling SVM decision boundary distance.

    Trains a linear SVM on a trailing window and returns the signed distance
    of each bar to the decision hyperplane.  Far from boundary = confident
    classification.  Retrains every ``period // 4`` bars.

    Parameters
    ----------
    features_matrix : (N, K) feature array.
    labels : (N,) binary labels (0/1 or -1/+1).
    period : training window length.

    Returns
    -------
    distance_to_boundary : (N,) signed distance.  NaN during warmup.
    """
    n, k = features_matrix.shape
    distance = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return distance

    retrain_every = max(1, period // 4)
    model: SVC | None = None
    mean_x: np.ndarray | None = None
    std_x: np.ndarray | None = None

    for i in range(period, n):
        need_train = model is None or (i - period) % retrain_every == 0

        if need_train:
            window_x = features_matrix[i - period : i]
            window_y = labels[i - period : i]
            valid = ~np.isnan(window_y) & ~np.any(np.isnan(window_x), axis=1)
            if np.sum(valid) < max(10, k + 1):
                continue
            X_t = window_x[valid]
            y_t = window_y[valid]
            # Need at least 2 classes
            unique = np.unique(y_t)
            if len(unique) < 2:
                continue
            std_w = np.std(X_t, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean_x = np.mean(X_t, axis=0)
            std_x = std_w
            X_t = (X_t - mean_x) / std_x
            model = SVC(kernel="linear", C=1.0)
            model.fit(X_t, y_t)

        if model is not None and mean_x is not None and std_x is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed = ((row - mean_x) / std_x).reshape(1, -1)
            distance[i] = model.decision_function(normed)[0]

    return distance
