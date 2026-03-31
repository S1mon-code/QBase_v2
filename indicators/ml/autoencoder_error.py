import numpy as np
from sklearn.decomposition import PCA


def reconstruction_error(
    features_matrix: np.ndarray,
    period: int = 120,
    encoding_dim: int = 2,
) -> np.ndarray:
    """Simplified autoencoder (PCA-based) reconstruction error.

    Projects each bar's feature vector down to ``encoding_dim`` dimensions
    via PCA, then reconstructs it.  The squared reconstruction error
    measures how unusual the current market state is relative to
    recent history.

    Parameters
    ----------
    features_matrix : (N, K) array of features.
    period : training window length.
    encoding_dim : number of latent dimensions.

    Returns
    -------
    reconstruction_error : (N,) squared reconstruction error.
        High error = unusual market state.
    """
    n, k = features_matrix.shape
    errors = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return errors

    encoding_dim = min(encoding_dim, k)
    retrain_every = max(1, period // 4)
    model: PCA | None = None
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
            model = PCA(n_components=encoding_dim)
            model.fit(normed)

        if model is not None and mean is not None and std is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed_row = ((row - mean) / std).reshape(1, -1)
            # Encode then decode
            encoded = model.transform(normed_row)
            decoded = model.inverse_transform(encoded)
            errors[i] = np.sum((normed_row[0] - decoded[0]) ** 2)

    return errors
