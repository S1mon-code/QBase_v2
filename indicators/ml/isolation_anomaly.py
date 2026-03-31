import numpy as np
from sklearn.ensemble import IsolationForest


def isolation_anomaly(
    features_matrix: np.ndarray,
    period: int = 120,
    contamination: float = 0.05,
) -> np.ndarray:
    """Rolling Isolation Forest anomaly score.

    Trains an Isolation Forest on a trailing window of ``period`` bars and
    scores the current bar.  Retrains every ``period // 4`` bars.

    Parameters
    ----------
    features_matrix : (N, K) array of features.
    period : training window length.
    contamination : expected fraction of anomalies.

    Returns
    -------
    anomaly_score : (N,) values in roughly (-1, 0).
        More negative = more anomalous.  NaN-padded during warmup.
    """
    n, k = features_matrix.shape
    scores = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return scores

    retrain_every = max(1, period // 4)
    model: IsolationForest | None = None
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
            model = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
            )
            model.fit(normed)

        if model is not None and mean is not None and std is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed_row = ((row - mean) / std).reshape(1, -1)
            # score_samples returns negative offset from threshold
            scores[i] = model.score_samples(normed_row)[0]

    return scores
