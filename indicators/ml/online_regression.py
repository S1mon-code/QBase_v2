import numpy as np


def online_sgd_signal(
    closes: np.ndarray,
    features_matrix: np.ndarray,
    learning_rate: float = 0.01,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Online SGD linear model that updates incrementally each bar.

    Maintains a linear weight vector updated via stochastic gradient
    descent on 1-bar return prediction.  Uses a trailing normalisation
    window of ``period`` bars for feature scaling.

    Parameters
    ----------
    closes : (N,) price series.
    features_matrix : (N, K) feature array.
    learning_rate : SGD step size.
    period : window for running normalisation statistics.

    Returns
    -------
    signal : (N,) model prediction for direction (positive = bullish).
    weights_norm : (N,) L2 norm of the weight vector (model confidence).
    """
    n = len(closes)
    k = features_matrix.shape[1]
    signal = np.full(n, np.nan, dtype=np.float64)
    weights_norm = np.full(n, np.nan, dtype=np.float64)

    if n < period + 2:
        return signal, weights_norm

    # 1-bar log returns
    safe = np.maximum(closes, 1e-12)
    log_p = np.log(safe)
    ret = np.full(n, np.nan, dtype=np.float64)
    ret[1:] = log_p[1:] - log_p[:-1]

    w = np.zeros(k, dtype=np.float64)

    for i in range(period + 1, n):
        # Normalise current features using trailing window
        window = features_matrix[i - period : i]
        if np.any(np.isnan(window)):
            continue
        mean_x = np.mean(window, axis=0)
        std_x = np.std(window, axis=0)
        std_x = np.where(std_x < 1e-12, 1.0, std_x)

        row = features_matrix[i]
        if np.any(np.isnan(row)):
            continue
        x_norm = (row - mean_x) / std_x

        # Predict
        pred = np.dot(w, x_norm)
        signal[i] = pred
        weights_norm[i] = np.linalg.norm(w)

        # Update using the realised return at this bar (known at bar close)
        y = ret[i]
        if np.isnan(y):
            continue
        error = y - pred
        # SGD step with L2 regularisation
        w = w * (1.0 - learning_rate * 0.01) + learning_rate * error * x_norm

    return signal, weights_norm
