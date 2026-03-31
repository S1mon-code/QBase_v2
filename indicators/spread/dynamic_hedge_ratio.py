"""Rolling OLS hedge ratio with Kalman-like exponential smoothing.

Tracks the time-varying relationship between two assets.
Changes in hedge ratio signal structural shifts in the pair.
"""

import numpy as np


def dynamic_hedge(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dynamic hedge ratio via rolling OLS with exponential smoothing.

    Parameters
    ----------
    closes_a : closing prices of asset A (dependent).
    closes_b : closing prices of asset B (independent).
    period   : rolling OLS window.

    Returns
    -------
    (hedge_ratio, hedge_ratio_change, spread)
        hedge_ratio        – smoothed rolling OLS beta (A = beta * B + alpha).
        hedge_ratio_change – first difference of hedge_ratio.
        spread             – A - hedge_ratio * B (residual spread).
    """
    n = len(closes_a)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()

    raw_beta = np.full(n, np.nan)

    for i in range(period - 1, n):
        ya = closes_a[i - period + 1 : i + 1].astype(float)
        xb = closes_b[i - period + 1 : i + 1].astype(float)
        mask = ~(np.isnan(ya) | np.isnan(xb))
        if np.sum(mask) < 10:
            continue

        ya_v = ya[mask]
        xb_v = xb[mask]
        x_mean = np.mean(xb_v)
        y_mean = np.mean(ya_v)
        denom = np.sum((xb_v - x_mean) ** 2)
        if denom == 0:
            continue
        raw_beta[i] = np.sum((xb_v - x_mean) * (ya_v - y_mean)) / denom

    # Exponential smoothing (Kalman-like)
    alpha = 2.0 / (period + 1)
    hedge_ratio = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(raw_beta[i]):
            continue
        if np.isnan(hedge_ratio[i - 1]) if i > 0 else True:
            hedge_ratio[i] = raw_beta[i]
        else:
            hedge_ratio[i] = alpha * raw_beta[i] + (1 - alpha) * hedge_ratio[i - 1]

    # Hedge ratio change
    hr_change = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(hedge_ratio[i]) and not np.isnan(hedge_ratio[i - 1]):
            hr_change[i] = hedge_ratio[i] - hedge_ratio[i - 1]

    # Spread = A - hedge_ratio * B
    spread = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(hedge_ratio[i]):
            spread[i] = closes_a[i] - hedge_ratio[i] * closes_b[i]

    return hedge_ratio, hr_change, spread
