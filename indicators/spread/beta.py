"""Rolling beta, alpha, and R-squared of an asset versus a benchmark.

Uses simple OLS regression on returns over a rolling window.
"""

import numpy as np


def rolling_beta(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling beta of asset vs benchmark.

    Parameters
    ----------
    asset_returns     : return series of the asset.
    benchmark_returns : return series of the benchmark.
    period            : rolling window size.

    Returns
    -------
    (beta, alpha, r_squared)
        beta      – rolling OLS slope (market sensitivity).
        alpha     – rolling OLS intercept (excess return).
        r_squared – rolling coefficient of determination.
    """
    n = len(asset_returns)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    beta = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    r_sq = np.full(n, np.nan)

    for i in range(period - 1, n):
        y = asset_returns[i - period + 1 : i + 1].astype(float)
        x = benchmark_returns[i - period + 1 : i + 1].astype(float)
        mask = ~(np.isnan(y) | np.isnan(x))
        if np.sum(mask) < 5:
            continue
        yv, xv = y[mask], x[mask]

        x_mean = np.mean(xv)
        y_mean = np.mean(yv)
        x_demean = xv - x_mean
        denom = np.dot(x_demean, x_demean)
        if denom == 0:
            continue

        b = np.dot(x_demean, yv - y_mean) / denom
        a = y_mean - b * x_mean
        beta[i] = b
        alpha[i] = a

        y_hat = a + b * xv
        ss_res = np.sum((yv - y_hat) ** 2)
        ss_tot = np.sum((yv - y_mean) ** 2)
        r_sq[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return beta, alpha, r_sq
