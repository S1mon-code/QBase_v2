import numpy as np
from sklearn.cluster import KMeans


def oi_cluster(closes: np.ndarray, oi: np.ndarray,
               volumes: np.ndarray, period: int = 120,
               n_clusters: int = 4) -> tuple:
    """K-means clustering on OI-price-volume state.

    Each cluster represents a distinct market state defined by the
    combination of OI level, OI change, price momentum, and volume.
    Retrained every period//4 bars.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volume.
    period : int
        Training window size.
    n_clusters : int
        Number of clusters (market states).

    Returns
    -------
    cluster_label : np.ndarray (int)
        Cluster assignment (0 to n_clusters-1). -1 during warmup.
    cluster_distance : np.ndarray
        Distance to assigned cluster centroid (lower = more typical).
    """
    n = len(closes)
    cluster_label = np.full(n, -1, dtype=int)
    cluster_distance = np.full(n, np.nan)

    if n < period + 1:
        return cluster_label, cluster_distance

    # Pre-compute features
    ret = np.full(n, 0.0)
    ret[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-10)

    oi_chg = np.full(n, 0.0)
    oi_chg[1:] = np.diff(oi)

    oi_chg_pct = np.full(n, 0.0)
    for i in range(1, n):
        if oi[i - 1] > 0:
            oi_chg_pct[i] = oi_chg[i] / oi[i - 1]

    vol_norm = np.full(n, 0.0)
    for i in range(n):
        if oi[i] > 0:
            vol_norm[i] = volumes[i] / oi[i]

    retrain_interval = max(period // 4, 1)
    model = None
    feat_mean = None
    feat_std = None

    for i in range(period, n):
        if model is None or (i - period) % retrain_interval == 0:
            idx_s = i - period
            idx_e = i

            X = np.column_stack([
                oi_chg_pct[idx_s:idx_e],
                ret[idx_s:idx_e],
                vol_norm[idx_s:idx_e],
            ])

            valid = ~np.any(np.isnan(X), axis=1)
            X_valid = X[valid]

            if len(X_valid) < n_clusters * 3:
                continue

            feat_mean = np.mean(X_valid, axis=0)
            feat_std = np.std(X_valid, axis=0)
            feat_std[feat_std == 0] = 1.0
            X_norm = (X_valid - feat_mean) / feat_std

            actual_k = min(n_clusters, len(X_norm))
            model = KMeans(
                n_clusters=actual_k, n_init=5,
                random_state=42, max_iter=100
            )
            model.fit(X_norm)

        if model is None or feat_mean is None:
            continue

        feat = np.array([
            oi_chg_pct[i], ret[i], vol_norm[i]
        ]).reshape(1, -1)
        feat_norm = (feat - feat_mean) / feat_std

        label = model.predict(feat_norm)[0]
        cluster_label[i] = label

        dist = np.linalg.norm(feat_norm - model.cluster_centers_[label])
        cluster_distance[i] = dist

    return cluster_label, cluster_distance
