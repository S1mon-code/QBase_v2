"""Price Efficiency Coefficient (PEC).

Measures how directly price moves from A to B.  PEC=1 means a
perfectly efficient (straight-line) move; PEC near 0 means noisy,
back-and-forth price action.
"""

import numpy as np


def price_efficiency_coefficient(
    closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Price Efficiency Coefficient: |net change| / sum(|changes|).

    Parameters
    ----------
    closes : array of closing prices.
    period : lookback window.

    Returns
    -------
    (pec, pec_smoothed)
        pec          – raw PEC, range [0, 1].
        pec_smoothed – exponentially smoothed PEC.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    pec = np.full(n, np.nan)

    for i in range(period, n):
        net_change = abs(closes[i] - closes[i - period])
        total_path = 0.0
        valid = True
        for j in range(i - period + 1, i + 1):
            if np.isnan(closes[j]) or np.isnan(closes[j - 1]):
                valid = False
                break
            total_path += abs(closes[j] - closes[j - 1])

        if valid and total_path > 0:
            pec[i] = net_change / total_path

    # Exponential smoothing
    alpha = 2.0 / (period + 1)
    smoothed = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(pec[i]):
            continue
        if np.isnan(smoothed[i - 1]) if i > 0 else True:
            smoothed[i] = pec[i]
        else:
            smoothed[i] = alpha * pec[i] + (1 - alpha) * smoothed[i - 1]

    return pec, smoothed
