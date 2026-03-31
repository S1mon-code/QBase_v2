import numpy as np
from sklearn.manifold import MDS


def manifold_embedding(features_matrix, period=120, n_components=2):
    """Rolling manifold embedding for regime visualisation.

    Fits sklearn MDS on a rolling window of the feature matrix and returns
    the embedding coordinates plus the distance from the window centroid
    (high distance = unusual regime).

    The model is retrained every ``period // 4`` bars to avoid excessive
    computation.

    Parameters
    ----------
    features_matrix : (N, K) array of features.
    period : rolling window size.
    n_components : embedding dimensionality.

    Returns
    -------
    embedding : (N, n_components) embedded coordinates.
    distance_from_center : (N,) Euclidean distance of the current point
        from the window centroid in embedded space.
    """
    features_matrix = np.asarray(features_matrix, dtype=np.float64)
    n, k = features_matrix.shape
    embedding = np.full((n, n_components), np.nan, dtype=np.float64)
    distance_from_center = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return embedding, distance_from_center

    retrain_interval = max(1, period // 4)
    last_model = None
    last_embed = None
    last_start = -1

    for i in range(period - 1, n):
        need_retrain = (last_model is None) or ((i - (period - 1)) % retrain_interval == 0)

        if need_retrain:
            win = features_matrix[i - period + 1: i + 1]
            if np.any(np.isnan(win)):
                # try dropping NaN rows
                mask = ~np.any(np.isnan(win), axis=1)
                if mask.sum() < max(n_components + 1, 5):
                    continue
                win_clean = win[mask]
            else:
                win_clean = win

            # standardise
            mu = win_clean.mean(axis=0)
            std = win_clean.std(axis=0)
            std[std < 1e-10] = 1.0
            win_norm = (win_clean - mu) / std

            try:
                mds = MDS(n_components=n_components, dissimilarity='euclidean',
                          random_state=42, max_iter=100, normalized_stress='auto')
                emb = mds.fit_transform(win_norm)
            except Exception:
                continue

            last_embed = emb
            last_mu = mu
            last_std = std
            last_start = i - period + 1

        if last_embed is None:
            continue

        # current point's position in the embedding
        idx_in_window = i - last_start
        if 0 <= idx_in_window < len(last_embed):
            pt = last_embed[idx_in_window]
            embedding[i] = pt
            centroid = last_embed.mean(axis=0)
            distance_from_center[i] = np.sqrt(np.sum((pt - centroid) ** 2))

    return embedding, distance_from_center
