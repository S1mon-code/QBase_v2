"""Momentum of the residual after removing factor exposure.

After hedging out the common factor, residual momentum captures
asset-specific alpha that is uncorrelated with the broad market.
"""

import numpy as np


def residual_momentum(
    asset_closes: np.ndarray,
    factor_closes: np.ndarray,
    period: int = 60,
    mom_period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Residual momentum: momentum of OLS residual vs factor.

    Parameters
    ----------
    asset_closes  : closing prices of the target asset.
    factor_closes : closing prices of the factor/benchmark.
    period        : rolling window for OLS regression.
    mom_period    : lookback for momentum of the residual.

    Returns
    -------
    (residual, residual_momentum_score)
        residual                – OLS residual (asset return - beta * factor return).
        residual_momentum_score – rolling sum of residual over mom_period.
    """
    n = len(asset_closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    ret_a = np.full(n, np.nan)
    ret_f = np.full(n, np.nan)
    for i in range(1, n):
        if asset_closes[i - 1] != 0 and not np.isnan(asset_closes[i - 1]):
            ret_a[i] = asset_closes[i] / asset_closes[i - 1] - 1.0
        if factor_closes[i - 1] != 0 and not np.isnan(factor_closes[i - 1]):
            ret_f[i] = factor_closes[i] / factor_closes[i - 1] - 1.0

    residual = np.full(n, np.nan)

    for i in range(period, n):
        wa = ret_a[i - period + 1 : i + 1]
        wf = ret_f[i - period + 1 : i + 1]
        mask = ~(np.isnan(wa) | np.isnan(wf))
        if np.sum(mask) < 10:
            continue

        ya = wa[mask]
        xf = wf[mask]

        x_mean = np.mean(xf)
        y_mean = np.mean(ya)
        denom = np.sum((xf - x_mean) ** 2)
        if denom == 0:
            continue

        beta = np.sum((xf - x_mean) * (ya - y_mean)) / denom
        residual[i] = ret_a[i] - beta * ret_f[i]

    # Residual momentum = rolling sum of residual
    res_mom = np.full(n, np.nan)
    for i in range(mom_period - 1, n):
        window = residual[i - mom_period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            res_mom[i] = np.sum(valid)

    return residual, res_mom
