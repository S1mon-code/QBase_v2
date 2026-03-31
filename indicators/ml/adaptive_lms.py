import numpy as np


def lms_filter(closes, reference=None, period=20, mu=0.01):
    """LMS (Least Mean Squares) adaptive filter.

    Adapts filter weights to minimise the error between the predicted signal
    and a reference.  When ``reference`` is None, uses ``closes`` lagged by 1
    as the reference (self-adaptive denoising).

    Parameters
    ----------
    closes : 1-D array of close prices.
    reference : 1-D array (same length) or None.
    period : filter order (number of taps).
    mu : step size / learning rate.

    Returns
    -------
    filtered_signal : (N,) filter output (prediction of reference).
    error : (N,) prediction error at each bar.
    weights_norm : (N,) L2-norm of filter weight vector (tracks adaptation).
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)

    if reference is None:
        ref = np.full(n, np.nan, dtype=np.float64)
        ref[1:] = closes[:-1]
    else:
        ref = np.asarray(reference, dtype=np.float64)

    filtered_signal = np.full(n, np.nan, dtype=np.float64)
    error = np.full(n, np.nan, dtype=np.float64)
    weights_norm = np.full(n, np.nan, dtype=np.float64)

    w = np.zeros(period, dtype=np.float64)

    for i in range(period - 1, n):
        if np.isnan(ref[i]):
            continue
        x = closes[i - period + 1: i + 1][::-1]  # most recent first
        if np.any(np.isnan(x)):
            continue

        y_hat = np.dot(w, x)
        e = ref[i] - y_hat

        # normalised LMS update (NLMS for stability)
        power = np.dot(x, x) + 1e-10
        w = w + (mu * e / power) * x

        filtered_signal[i] = y_hat
        error[i] = e
        weights_norm[i] = np.sqrt(np.dot(w, w))

    return filtered_signal, error, weights_norm
