"""Adverse selection component — how much does price move against order flow.

Estimates the information asymmetry component of the spread by
measuring how much prices move in the direction of trades.
High adverse selection = informed traders dominating.
"""

import numpy as np


def adverse_selection(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Adverse selection: price impact in direction of volume-classified trades.

    Parameters
    ----------
    closes  : array of closing prices.
    volumes : array of trading volumes.
    period  : rolling window.

    Returns
    -------
    (as_score, as_zscore)
        as_score  – average signed price impact weighted by volume.
                    Higher = more adverse selection (informed trading).
        as_zscore – rolling z-score of as_score.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Compute signed impact: return_t * sign(return_{t-1}) * volume_t
    # If price continues in the same direction as previous move, informed traders
    # are pushing price further → adverse selection
    impact = np.full(n, np.nan)
    for i in range(2, n):
        if (np.isnan(closes[i]) or np.isnan(closes[i - 1])
                or np.isnan(closes[i - 2]) or np.isnan(volumes[i])):
            continue
        if closes[i - 1] == 0 or closes[i - 2] == 0:
            continue

        ret_curr = closes[i] / closes[i - 1] - 1.0
        ret_prev = closes[i - 1] / closes[i - 2] - 1.0

        # Trade sign from previous return direction
        if ret_prev > 0:
            sign = 1.0
        elif ret_prev < 0:
            sign = -1.0
        else:
            sign = 0.0

        # Adverse selection = how much price continues in trade direction
        impact[i] = ret_curr * sign * np.log1p(volumes[i])

    as_score = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = impact[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            as_score[i] = np.mean(valid)

    # Z-score
    zperiod = max(period * 3, 60)
    zscore = np.full(n, np.nan)
    for i in range(zperiod - 1, n):
        window = as_score[i - zperiod + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            zscore[i] = (as_score[i] - mu) / sigma

    return as_score, zscore
