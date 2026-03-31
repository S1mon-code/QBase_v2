"""Kyle's lambda: price impact per unit volume.

A regression-based measure of market depth.  Higher lambda means
each unit of volume moves prices more (less liquid).
"""

import numpy as np


def kyle_lambda(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Kyle's lambda: price impact per unit volume.

    Estimated by regressing absolute price returns on signed volume
    (volume * sign(return)) over a rolling window.

    Parameters
    ----------
    closes  : closing prices.
    volumes : trading volumes.
    period  : rolling window for OLS regression.

    Returns
    -------
    (lambda_val, lambda_zscore)
        lambda_val    – rolling Kyle's lambda (price impact coefficient).
        lambda_zscore – rolling z-score of lambda.  High values signal
                        deteriorating market depth.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if n < 3:
        return np.full(n, np.nan), np.full(n, np.nan)

    # Returns and signed volume
    rets = np.full(n, np.nan)
    signed_vol = np.full(n, np.nan)
    for i in range(1, n):
        if closes[i - 1] != 0 and not np.isnan(closes[i - 1]):
            r = closes[i] / closes[i - 1] - 1.0
            rets[i] = r
            if not np.isnan(volumes[i]) and volumes[i] > 0:
                signed_vol[i] = volumes[i] * np.sign(r) if r != 0 else 0.0

    lam = np.full(n, np.nan)
    lam_z = np.full(n, np.nan)

    for i in range(period, n):
        y = rets[i - period + 1 : i + 1]
        x = signed_vol[i - period + 1 : i + 1]
        mask = ~(np.isnan(y) | np.isnan(x))
        if np.sum(mask) < 10:
            continue
        yv, xv = y[mask], x[mask]

        x_mean = np.mean(xv)
        x_demean = xv - x_mean
        denom = np.dot(x_demean, x_demean)
        if denom == 0:
            continue

        beta = np.dot(x_demean, yv - np.mean(yv)) / denom
        lam[i] = abs(beta)

    # Z-score
    zperiod = max(period, 60)
    for i in range(zperiod - 1, n):
        window = lam[i - zperiod + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            lam_z[i] = (lam[i] - mu) / sigma

    return lam, lam_z
