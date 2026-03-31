import numpy as np
from sklearn.decomposition import IncrementalPCA


def incremental_pca_signal(
    features_matrix: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """True incremental PCA that updates without full retrain.

    Uses sklearn's IncrementalPCA with partial_fit, processing data in
    batches.  After initial fit, each new batch updates the model
    incrementally.

    Parameters
    ----------
    features_matrix : (N, K) feature array.
    n_components : number of principal components to keep.

    Returns
    -------
    components : (N, n_components) projected features.
    explained_var_ratio : (N, n_components) explained variance ratios.
    """
    n, k = features_matrix.shape
    n_comp = min(n_components, k)
    components = np.full((n, n_comp), np.nan, dtype=np.float64)
    explained = np.full((n, n_comp), np.nan, dtype=np.float64)

    if n < n_comp * 2:
        return components, explained

    batch_size = max(n_comp * 2, 30)
    model = IncrementalPCA(n_components=n_comp)

    i = 0
    while i < n:
        end = min(i + batch_size, n)
        batch = features_matrix[i:end]

        # Skip batches with NaN
        if np.any(np.isnan(batch)):
            i = end
            continue

        # Need at least n_comp samples for partial_fit
        if (end - i) < n_comp:
            if model.n_samples_seen_ is not None and model.n_samples_seen_ > 0:
                # Transform remaining with existing model
                try:
                    proj = model.transform(batch)
                    components[i:end] = proj
                    explained[i:end] = model.explained_variance_ratio_[:n_comp]
                except Exception:
                    pass
            i = end
            continue

        model.partial_fit(batch)
        proj = model.transform(batch)
        components[i:end] = proj
        explained[i:end] = model.explained_variance_ratio_[:n_comp]
        i = end

    return components, explained
