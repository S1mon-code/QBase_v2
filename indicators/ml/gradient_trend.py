import numpy as np


def gradient_signal(
    closes: np.ndarray,
    period: int = 20,
    smoothing: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth gradient-based trend signal (first and second derivatives).

    Computes smoothed first, second, and third derivatives of the price
    series using numpy convolution with a Gaussian-like SMA kernel.

    Parameters
    ----------
    closes : (N,) price series.
    period : window for gradient estimation via linear regression slope.
    smoothing : additional SMA smoothing applied to derivatives.

    Returns
    -------
    gradient : (N,) first derivative — direction of trend.
    acceleration : (N,) second derivative — momentum change.
    jerk : (N,) third derivative — acceleration change.
    """
    n = len(closes)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    gradient = np.full(n, np.nan, dtype=np.float64)
    acceleration = np.full(n, np.nan, dtype=np.float64)
    jerk = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return gradient, acceleration, jerk

    # First derivative: rolling linear regression slope
    x = np.arange(period, dtype=np.float64)
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        if np.any(np.isnan(window)):
            continue
        y_mean = np.mean(window)
        gradient[i] = np.sum((x - x_mean) * (window - y_mean)) / x_var

    # Smooth the gradient
    gradient = _sma(gradient, smoothing)

    # Second derivative: diff of gradient, then smooth
    raw_accel = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(gradient[i]) and not np.isnan(gradient[i - 1]):
            raw_accel[i] = gradient[i] - gradient[i - 1]
    acceleration = _sma(raw_accel, smoothing)

    # Third derivative: diff of acceleration, then smooth
    raw_jerk = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(acceleration[i]) and not np.isnan(acceleration[i - 1]):
            raw_jerk[i] = acceleration[i] - acceleration[i - 1]
    jerk = _sma(raw_jerk, smoothing)

    return gradient, acceleration, jerk


def _sma(data: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average preserving NaN structure."""
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 1:
        return data.copy()
    for i in range(window - 1, n):
        segment = data[i - window + 1 : i + 1]
        valid = segment[~np.isnan(segment)]
        if len(valid) == window:
            out[i] = np.mean(valid)
    return out
