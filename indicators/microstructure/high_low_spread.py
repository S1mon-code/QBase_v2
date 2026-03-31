"""Corwin-Schultz (2012) high-low spread estimator.

Estimates the bid-ask spread from daily high and low prices.
Based on the insight that daily highs are more likely at ask
and lows at bid prices.
"""

import numpy as np


def hl_spread(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Corwin-Schultz high-low spread estimator.

    Parameters
    ----------
    highs  : array of high prices.
    lows   : array of low prices.
    closes : array of closing prices (for percentage calculation).
    period : smoothing window.

    Returns
    -------
    (spread_est, spread_pct)
        spread_est – estimated bid-ask spread in price units.
        spread_pct – spread as percentage of closing price.
    """
    n = len(highs)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    spread_est = np.full(n, np.nan)
    spread_pct = np.full(n, np.nan)

    ln2 = np.log(2.0)
    sqrt2 = np.sqrt(2.0)

    for i in range(1, n):
        if (np.isnan(highs[i]) or np.isnan(lows[i]) or
                np.isnan(highs[i - 1]) or np.isnan(lows[i - 1])):
            continue
        if lows[i] <= 0 or lows[i - 1] <= 0:
            continue

        # Beta = sum of squared log(H/L) over 2 consecutive bars
        beta = (np.log(highs[i] / lows[i])) ** 2 + (np.log(highs[i - 1] / lows[i - 1])) ** 2

        # Gamma = log(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))^2
        gamma = (np.log(max(highs[i], highs[i - 1]) / min(lows[i], lows[i - 1]))) ** 2

        # Alpha
        denom = 3.0 - 2.0 * sqrt2
        if denom == 0:
            continue
        alpha_val = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)

        # Spread = 2 * (e^alpha - 1) / (1 + e^alpha)
        if alpha_val > 0:
            ea = np.exp(alpha_val)
            s = 2.0 * (ea - 1.0) / (1.0 + ea)
        else:
            s = 0.0

        spread_est[i] = s * closes[i] if not np.isnan(closes[i]) else np.nan
        if closes[i] > 0 and not np.isnan(closes[i]):
            spread_pct[i] = s * 100.0

    # Smooth with rolling average
    smooth_est = np.full(n, np.nan)
    smooth_pct = np.full(n, np.nan)
    for i in range(period - 1, n):
        w_est = spread_est[i - period + 1 : i + 1]
        w_pct = spread_pct[i - period + 1 : i + 1]
        v_est = w_est[~np.isnan(w_est)]
        v_pct = w_pct[~np.isnan(w_pct)]
        if len(v_est) > 0:
            smooth_est[i] = np.mean(v_est)
        if len(v_pct) > 0:
            smooth_pct[i] = np.mean(v_pct)

    return smooth_est, smooth_pct
