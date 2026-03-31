"""Realized spread proxy from serial correlation of returns.

Estimates the effective bid-ask spread using the Roll (1984) model
extended with a rolling window.  Higher values indicate wider
effective spreads and lower liquidity.
"""

import numpy as np


def realized_spread(
    closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Realized spread from serial covariance of returns.

    Parameters
    ----------
    closes : array of closing prices.
    period : rolling window for estimation.

    Returns
    -------
    (r_spread, r_spread_zscore)
        r_spread        – estimated effective spread (2 * sqrt(-cov) if cov < 0).
        r_spread_zscore – rolling z-score of spread estimate.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Log returns
    ret = np.full(n, np.nan)
    for i in range(1, n):
        if closes[i - 1] > 0 and not np.isnan(closes[i]) and not np.isnan(closes[i - 1]):
            ret[i] = np.log(closes[i] / closes[i - 1])

    r_spread = np.full(n, np.nan)
    for i in range(period + 1, n):
        r1 = ret[i - period + 1 : i + 1]
        r0 = ret[i - period : i]
        mask = ~(np.isnan(r1) | np.isnan(r0))
        if np.sum(mask) < 5:
            continue

        cov = np.mean(r1[mask] * r0[mask]) - np.mean(r1[mask]) * np.mean(r0[mask])
        if cov < 0:
            r_spread[i] = 2.0 * np.sqrt(-cov)
        else:
            r_spread[i] = 0.0

    # Z-score
    zperiod = max(period * 3, 60)
    zscore = np.full(n, np.nan)
    for i in range(zperiod - 1, n):
        window = r_spread[i - zperiod + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            zscore[i] = (r_spread[i] - mu) / sigma

    return r_spread, zscore
