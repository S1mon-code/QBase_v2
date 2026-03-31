import numpy as np


def distance_correlation(
    series_a: np.ndarray,
    series_b: np.ndarray,
    period: int = 60,
) -> np.ndarray:
    """Rolling distance correlation between two series.

    Distance correlation captures nonlinear dependence (unlike Pearson).
    dcor = 0 implies independence; dcor = 1 implies strong dependence.

    Parameters
    ----------
    series_a : (N,) first series.
    series_b : (N,) second series.
    period : rolling window length.

    Returns
    -------
    dcor : (N,) distance correlation (0 to 1).
    """
    n = len(series_a)
    dcor = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return dcor

    for i in range(period, n):
        a = series_a[i - period : i]
        b = series_b[i - period : i]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue
        dcor[i] = _dcor(a, b)

    return dcor


def _dcor(x: np.ndarray, y: np.ndarray) -> float:
    """Compute distance correlation between two 1D arrays."""
    n = len(x)
    if n < 3:
        return np.nan

    # Distance matrices
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    # Double centring
    a_row = np.mean(a, axis=1, keepdims=True)
    a_col = np.mean(a, axis=0, keepdims=True)
    a_grand = np.mean(a)
    A = a - a_row - a_col + a_grand

    b_row = np.mean(b, axis=1, keepdims=True)
    b_col = np.mean(b, axis=0, keepdims=True)
    b_grand = np.mean(b)
    B = b - b_row - b_col + b_grand

    dcov2_xy = np.mean(A * B)
    dcov2_xx = np.mean(A * A)
    dcov2_yy = np.mean(B * B)

    denom = np.sqrt(dcov2_xx * dcov2_yy)
    if denom < 1e-15:
        return 0.0

    dcor2 = dcov2_xy / denom
    # Can be slightly negative due to numerics
    return np.sqrt(max(0.0, dcor2))
