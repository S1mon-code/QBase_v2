import numpy as np
from sklearn.cluster import SpectralClustering


def spectral_regime(
    features_matrix: np.ndarray,
    period: int = 120,
    n_clusters: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral clustering for regime detection.

    Uses spectral clustering on a trailing window to capture non-linear
    regime structure via the affinity graph.  Retrains every ``period // 4``
    bars; between retrains, assigns new bars to the nearest centroid of
    the last clustering.

    Parameters
    ----------
    features_matrix : (N, K) array of features.
    period : training window length.
    n_clusters : number of regimes.

    Returns
    -------
    labels : (N,) regime label.  NaN-padded at start.
    affinity_score : (N,) mean affinity of the current bar to its cluster
        members (higher = more typical of that regime).
    """
    n, k = features_matrix.shape
    labels = np.full(n, np.nan, dtype=np.float64)
    affinity = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return labels, affinity

    retrain_every = max(1, period // 4)
    centroids: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    for i in range(period, n):
        need_train = centroids is None or (i - period) % retrain_every == 0

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

            try:
                sc = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="rbf",
                    random_state=42,
                    n_init=3,
                )
                cluster_labels = sc.fit_predict(normed)
            except Exception:
                continue

            # Compute centroids for prediction between retrains
            centroids = np.zeros((n_clusters, k), dtype=np.float64)
            for c in range(n_clusters):
                mask = cluster_labels == c
                if np.any(mask):
                    centroids[c] = np.mean(normed[mask], axis=0)

        if centroids is not None and mean is not None and std is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed_row = (row - mean) / std

            # Assign to nearest centroid
            dists = np.array([np.linalg.norm(normed_row - c) for c in centroids])
            lbl = int(np.argmin(dists))
            labels[i] = lbl

            # Affinity: inverse distance (higher = closer to centroid)
            d = dists[lbl]
            affinity[i] = np.exp(-d ** 2 / (2.0 * k))  # RBF-like score

    return labels, affinity
