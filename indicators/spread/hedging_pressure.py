"""Hedging pressure proxy from open interest and price direction.

Uses the relationship between OI changes and price movement to infer
whether hedgers (commercial participants) are actively positioning.
"""

import numpy as np


def hedging_pressure(
    closes: np.ndarray,
    oi: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Hedging pressure proxy.

    When OI rises sharply while prices move against the prevailing
    trend (or are flat), it suggests hedgers are entering positions
    to lock in prices.

    Parameters
    ----------
    closes  : closing prices.
    oi      : open interest series.
    volumes : volume series.
    period  : lookback window for smoothing.

    Returns
    -------
    np.ndarray
        Hedging pressure score.  Higher values indicate more hedging
        activity (commercial participants taking contra-trend positions).
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float)
    if n < 3:
        return np.full(n, np.nan)

    hp = np.full(n, np.nan)

    # Price return and OI change (1-bar)
    price_ret = np.full(n, np.nan)
    oi_chg = np.full(n, np.nan)
    for i in range(1, n):
        if closes[i - 1] != 0 and not np.isnan(closes[i - 1]):
            price_ret[i] = closes[i] / closes[i - 1] - 1.0
        if oi[i - 1] != 0 and not np.isnan(oi[i - 1]):
            oi_chg[i] = oi[i] / oi[i - 1] - 1.0

    for i in range(period, n):
        pr = price_ret[i - period + 1 : i + 1]
        oc = oi_chg[i - period + 1 : i + 1]
        mask = ~(np.isnan(pr) | np.isnan(oc))
        if np.sum(mask) < 5:
            continue
        prv, ocv = pr[mask], oc[mask]

        # Hedging pressure = OI growth when price moves against trend
        # Score: mean(OI_chg * -sign(price_ret))
        hp[i] = np.mean(ocv * -np.sign(prv))

    return hp
