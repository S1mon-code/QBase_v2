"""Tick direction indicator — rolling fraction of upticks vs downticks.

Provides a simple measure of price direction persistence that is
sensitive to microstructure effects like momentum ignition.
"""

import numpy as np


def tick_direction(
    closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Tick direction: fraction of upticks over rolling window.

    Parameters
    ----------
    closes : array of closing prices.
    period : rolling window.

    Returns
    -------
    (tick_ratio, tick_momentum)
        tick_ratio    – fraction of upticks in window, range [0, 1].
                        0.5 = neutral, >0.5 = bullish ticks dominate.
        tick_momentum – rate of change of tick_ratio.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Classify ticks
    ticks = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(closes[i]) or np.isnan(closes[i - 1]):
            continue
        if closes[i] > closes[i - 1]:
            ticks[i] = 1.0
        elif closes[i] < closes[i - 1]:
            ticks[i] = 0.0
        else:
            ticks[i] = 0.5  # unchanged

    tick_ratio = np.full(n, np.nan)
    for i in range(period, n):
        window = ticks[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            tick_ratio[i] = np.mean(valid)

    # Momentum of tick ratio
    mom_period = max(period // 2, 5)
    tick_mom = np.full(n, np.nan)
    for i in range(mom_period, n):
        if not np.isnan(tick_ratio[i]) and not np.isnan(tick_ratio[i - mom_period]):
            tick_mom[i] = tick_ratio[i] - tick_ratio[i - mom_period]

    return tick_ratio, tick_mom
