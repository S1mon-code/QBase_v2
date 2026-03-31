import numpy as np
from sklearn.mixture import BayesianGaussianMixture


def variational_regime(
    closes: np.ndarray,
    period: int = 120,
    n_components: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variational Bayesian GMM regime detection.

    Uses a Bayesian GMM that auto-detects the effective number of components
    (prunes unused ones).  Retrains every ``period // 4`` bars.

    Parameters
    ----------
    closes : (N,) price series.
    period : training window length.
    n_components : maximum number of mixture components.

    Returns
    -------
    labels : (N,) most likely component label.
    probabilities : (N,) probability of the assigned label.
    effective_components : (N,) number of active components (weight > 0.05).
    """
    n = len(closes)
    labels = np.full(n, np.nan, dtype=np.float64)
    probabilities = np.full(n, np.nan, dtype=np.float64)
    effective_comp = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return labels, probabilities, effective_comp

    # Compute log returns + rolling std as 2D feature
    safe = np.maximum(closes, 1e-12)
    log_ret = np.full(n, np.nan, dtype=np.float64)
    log_ret[1:] = np.log(safe[1:]) - np.log(safe[:-1])

    retrain_every = max(1, period // 4)
    model: BayesianGaussianMixture | None = None
    mean_x: np.ndarray | None = None
    std_x: np.ndarray | None = None
    n_eff: float = 0.0

    for i in range(period, n):
        need_train = model is None or (i - period) % retrain_every == 0

        if need_train:
            window = log_ret[i - period : i]
            if np.any(np.isnan(window)):
                continue

            # Build 2D features: [return, abs_return]
            X_train = np.column_stack([window, np.abs(window)])
            std_w = np.std(X_train, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean_x = np.mean(X_train, axis=0)
            std_x = std_w
            X_t = (X_train - mean_x) / std_x

            model = BayesianGaussianMixture(
                n_components=n_components,
                covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                random_state=42,
                max_iter=100,
            )
            model.fit(X_t)
            n_eff = float(np.sum(model.weights_ > 0.05))

        if model is not None and mean_x is not None and std_x is not None:
            ret_i = log_ret[i]
            if np.isnan(ret_i):
                continue
            row = np.array([[ret_i, abs(ret_i)]])
            normed = (row - mean_x) / std_x
            lbl = model.predict(normed)[0]
            prob = model.predict_proba(normed)[0]
            labels[i] = float(lbl)
            probabilities[i] = prob[lbl]
            effective_comp[i] = n_eff

    return labels, probabilities, effective_comp
