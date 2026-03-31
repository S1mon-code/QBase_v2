import numpy as np


def von_neumann_ratio(
    data: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Von Neumann ratio: variance of successive differences / variance.

    VN ~ 2 indicates random walk (no autocorrelation).
    VN < 2 suggests positive autocorrelation (trending).
    VN > 2 suggests negative autocorrelation (mean-reverting).

    Parameters
    ----------
    data : (N,) input series (e.g., returns or prices).
    period : rolling window length.

    Returns
    -------
    vn_ratio : (N,) Von Neumann ratio.
    is_random : (N,) 1.0 if |VN - 2| < 0.3 (approximately random), else 0.0.
    """
    n = len(data)
    vn_ratio = np.full(n, np.nan, dtype=np.float64)
    is_random = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return vn_ratio, is_random

    for i in range(period, n):
        window = data[i - period : i]
        if np.any(np.isnan(window)):
            continue

        var = np.var(window)
        if var < 1e-15:
            continue

        # Mean squared successive difference
        diffs = window[1:] - window[:-1]
        mssd = np.mean(diffs ** 2)

        vn = mssd / var
        vn_ratio[i] = vn
        is_random[i] = 1.0 if abs(vn - 2.0) < 0.3 else 0.0

    return vn_ratio, is_random
