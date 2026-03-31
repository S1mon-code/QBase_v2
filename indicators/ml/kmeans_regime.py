import numpy as np
from sklearn.cluster import KMeans


def kmeans_regime(
    features_matrix: np.ndarray,
    period: int = 120,
    n_clusters: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling K-means clustering for regime detection.

    Fits K-means on a trailing window of ``period`` bars and assigns the
    current bar to the nearest cluster.  Retrains every ``period // 4``
    bars to keep computational cost manageable.

    Parameters
    ----------
    features_matrix : (N, K) array of features.
    period : training window length.
    n_clusters : number of clusters.

    Returns
    -------
    labels : (N,) cluster label 0, 1, 2, …  NaN-padded at start.
    distances_to_center : (N,) Euclidean distance to the assigned centroid.
    """
    n, k = features_matrix.shape
    labels = np.full(n, np.nan, dtype=np.float64)
    distances = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return labels, distances

    retrain_every = max(1, period // 4)
    model: KMeans | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    for i in range(period, n):
        need_train = model is None or (i - period) % retrain_every == 0

        if need_train:
            window = features_matrix[i - period : i]
            if np.any(np.isnan(window)):
                continue
            std_w = np.std(window, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean = np.mean(window, axis=0)
            std = std_w
            normed = (window - mean) / std
            model = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
            model.fit(normed)

        if model is not None and mean is not None and std is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed_row = ((row - mean) / std).reshape(1, -1)
            lbl = model.predict(normed_row)[0]
            labels[i] = lbl
            center = model.cluster_centers_[lbl]
            distances[i] = np.linalg.norm(normed_row[0] - center)

    return labels, distances
