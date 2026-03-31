import numpy as np
from sklearn.mixture import GaussianMixture


def gmm_regime(
    features_matrix: np.ndarray,
    period: int = 120,
    n_components: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Gaussian Mixture Model regime classification.

    Fits a GMM on a trailing window and classifies the current bar into
    one of ``n_components`` regimes.  Retrains every ``period // 4`` bars.

    Parameters
    ----------
    features_matrix : (N, K) array of features.
    period : training window length.
    n_components : number of Gaussian components / regimes.

    Returns
    -------
    labels : (N,) most probable regime label.  NaN-padded at start.
    probabilities : (N, n_components) posterior probabilities per regime.
    """
    n, k = features_matrix.shape
    labels = np.full(n, np.nan, dtype=np.float64)
    probabilities = np.full((n, n_components), np.nan, dtype=np.float64)

    if n < period:
        return labels, probabilities

    retrain_every = max(1, period // 4)
    model: GaussianMixture | None = None
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
            model = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=42,
                max_iter=200,
            )
            model.fit(normed)

        if model is not None and mean is not None and std is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed_row = ((row - mean) / std).reshape(1, -1)
            labels[i] = model.predict(normed_row)[0]
            probabilities[i] = model.predict_proba(normed_row)[0]

    return labels, probabilities
