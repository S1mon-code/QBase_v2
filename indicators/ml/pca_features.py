import numpy as np
from sklearn.decomposition import PCA


def rolling_pca(
    features_matrix: np.ndarray,
    period: int = 60,
    n_components: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling PCA on multi-feature matrix.

    Reduces a (N, K) feature matrix to its top principal components using
    a rolling window.  The model is retrained every ``period // 4`` bars;
    between retrains the last fitted model is reused.

    Parameters
    ----------
    features_matrix : (N, K) array of raw features.
    period : training window length.
    n_components : number of principal components to keep.

    Returns
    -------
    components : (N, n_components) — projected features.
    explained_ratio : (N, n_components) — variance explained per component.
    """
    n, k = features_matrix.shape
    n_components = min(n_components, k)
    components = np.full((n, n_components), np.nan, dtype=np.float64)
    explained_ratio = np.full((n, n_components), np.nan, dtype=np.float64)

    if n < period:
        return components, explained_ratio

    retrain_every = max(1, period // 4)
    model: PCA | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    for i in range(period, n):
        need_train = model is None or (i - period) % retrain_every == 0

        if need_train:
            window = features_matrix[i - period : i]
            # skip if any column is constant or has NaNs
            if np.any(np.isnan(window)):
                continue
            std_w = np.std(window, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean = np.mean(window, axis=0)
            std = std_w
            normed = (window - mean) / std
            model = PCA(n_components=n_components)
            model.fit(normed)

        if model is not None and mean is not None and std is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed_row = ((row - mean) / std).reshape(1, -1)
            components[i] = model.transform(normed_row)[0]
            explained_ratio[i] = model.explained_variance_ratio_

    return components, explained_ratio
