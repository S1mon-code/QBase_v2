"""Rolling cointegration residual using OLS hedge ratio.

Computes a rolling OLS regression of asset A on asset B, then derives
the residual and its z-score for mean-reversion signals.
"""

import numpy as np


def cointegration_residual(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    period: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling cointegration residual.

    Parameters
    ----------
    closes_a : close prices of asset A (dependent variable).
    closes_b : close prices of asset B (independent variable).
    period   : rolling window for OLS regression.

    Returns
    -------
    (residual, zscore, hedge_ratio)
        residual    – A - hedge_ratio * B (the spread).
        zscore      – rolling z-score of the residual.
        hedge_ratio – rolling OLS slope (beta).
    """
    n = len(closes_a)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    residual = np.full(n, np.nan)
    zscore = np.full(n, np.nan)
    hedge_ratio = np.full(n, np.nan)

    for i in range(period - 1, n):
        y = closes_a[i - period + 1 : i + 1].astype(float)
        x = closes_b[i - period + 1 : i + 1].astype(float)

        mask = ~(np.isnan(y) | np.isnan(x))
        if np.sum(mask) < 10:
            continue

        y_v = y[mask]
        x_v = x[mask]

        # OLS: y = beta * x + alpha
        x_mean = np.mean(x_v)
        y_mean = np.mean(y_v)
        x_demean = x_v - x_mean
        denom = np.dot(x_demean, x_demean)
        if denom == 0:
            continue

        beta = np.dot(x_demean, y_v - y_mean) / denom
        hedge_ratio[i] = beta

        # Residuals over the window
        resid = y_v - beta * x_v
        mu = np.mean(resid)
        sigma = np.std(resid, ddof=1)

        cur_resid = closes_a[i] - beta * closes_b[i]
        residual[i] = cur_resid

        if sigma > 0:
            zscore[i] = (cur_resid - mu) / sigma

    return residual, zscore, hedge_ratio
