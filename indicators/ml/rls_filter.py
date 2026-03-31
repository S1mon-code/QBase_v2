import numpy as np


def rls_filter(closes, order=5, forgetting=0.99):
    """RLS (Recursive Least Squares) adaptive filter for trend extraction.

    Uses an inverse correlation matrix update (standard RLS) to predict the
    next close from the previous ``order`` closes.

    Parameters
    ----------
    closes : 1-D array of close prices.
    order : filter order (number of past samples used).
    forgetting : forgetting factor lambda in (0, 1].  Lower = faster adaptation.

    Returns
    -------
    prediction : (N,) one-step-ahead prediction.
    error : (N,) prediction error (actual - predicted).
    filter_gain : (N,) norm of the Kalman gain vector (tracks adaptation speed).
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    prediction = np.full(n, np.nan, dtype=np.float64)
    error = np.full(n, np.nan, dtype=np.float64)
    filter_gain = np.full(n, np.nan, dtype=np.float64)

    if n < order + 1:
        return prediction, error, filter_gain

    # initialise
    delta = 100.0  # large initial value for P
    P = delta * np.eye(order, dtype=np.float64)
    w = np.zeros(order, dtype=np.float64)
    lam = forgetting
    inv_lam = 1.0 / lam

    for i in range(order, n):
        x = closes[i - order: i][::-1]  # most recent first
        if np.any(np.isnan(x)) or np.isnan(closes[i]):
            continue

        # prediction
        y_hat = np.dot(w, x)
        prediction[i] = y_hat

        e = closes[i] - y_hat
        error[i] = e

        # gain vector
        Px = P @ x
        denom = lam + np.dot(x, Px)
        if abs(denom) < 1e-15:
            filter_gain[i] = 0.0
            continue
        k = Px / denom
        filter_gain[i] = np.sqrt(np.dot(k, k))

        # update weights
        w = w + k * e

        # update inverse correlation matrix
        P = inv_lam * (P - np.outer(k, np.dot(x, P)))

        # numerical stability: symmetrise
        P = 0.5 * (P + P.T)

    return prediction, error, filter_gain
