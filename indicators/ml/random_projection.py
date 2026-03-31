import numpy as np
from sklearn.random_projection import GaussianRandomProjection


def random_projection_features(
    features_matrix: np.ndarray,
    n_components: int = 5,
    period: int = 120,
) -> np.ndarray:
    """Random projection dimensionality reduction.

    Fast alternative to PCA.  Projects features into a lower-dimensional
    random subspace.  Retrains the projection matrix every ``period // 4``
    bars.

    Parameters
    ----------
    features_matrix : (N, K) feature array.
    n_components : target dimensionality.
    period : window used for fitting the projection.

    Returns
    -------
    projected : (N, n_components) projected features.  NaN during warmup.
    """
    n, k = features_matrix.shape
    n_comp = min(n_components, k)
    projected = np.full((n, n_comp), np.nan, dtype=np.float64)

    if n < period:
        return projected

    retrain_every = max(1, period // 4)
    model: GaussianRandomProjection | None = None
    mean_x: np.ndarray | None = None
    std_x: np.ndarray | None = None

    for i in range(period, n):
        need_train = model is None or (i - period) % retrain_every == 0

        if need_train:
            window = features_matrix[i - period : i]
            if np.any(np.isnan(window)):
                continue
            std_w = np.std(window, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean_x = np.mean(window, axis=0)
            std_x = std_w
            normed = (window - mean_x) / std_x
            model = GaussianRandomProjection(n_components=n_comp, random_state=42)
            model.fit(normed)

        if model is not None and mean_x is not None and std_x is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed = ((row - mean_x) / std_x).reshape(1, -1)
            projected[i] = model.transform(normed)[0]

    return projected
