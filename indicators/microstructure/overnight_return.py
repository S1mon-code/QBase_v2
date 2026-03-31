"""Overnight vs intraday return decomposition.

Splits total returns into the overnight gap component (prior close to
current open) and the intraday component (open to close).
"""

import numpy as np


def overnight_return(
    opens: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose returns into overnight and intraday components.

    Parameters
    ----------
    opens  : opening prices.
    closes : closing prices.
    period : lookback for rolling averages.

    Returns
    -------
    (overnight_ret, intraday_ret, overnight_ratio)
        overnight_ret  – close_{t-1} to open_t return.
        intraday_ret   – open_t to close_t return.
        overnight_ratio – rolling ratio: |avg overnight| /
                          (|avg overnight| + |avg intraday|).
                          High ratio means most movement happens
                          during the overnight session.
    """
    n = len(opens)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    overnight_ret = np.full(n, np.nan)
    intraday_ret = np.full(n, np.nan)
    overnight_ratio = np.full(n, np.nan)

    for i in range(1, n):
        prev_c = closes[i - 1]
        cur_o = opens[i]
        cur_c = closes[i]

        if prev_c != 0 and not np.isnan(prev_c) and not np.isnan(cur_o):
            overnight_ret[i] = cur_o / prev_c - 1.0
        if cur_o != 0 and not np.isnan(cur_o) and not np.isnan(cur_c):
            intraday_ret[i] = cur_c / cur_o - 1.0

    for i in range(period, n):
        on = overnight_ret[i - period + 1 : i + 1]
        intra = intraday_ret[i - period + 1 : i + 1]
        mask = ~(np.isnan(on) | np.isnan(intra))
        if np.sum(mask) < 3:
            continue
        avg_on = np.mean(np.abs(on[mask]))
        avg_intra = np.mean(np.abs(intra[mask]))
        total = avg_on + avg_intra
        if total > 0:
            overnight_ratio[i] = avg_on / total

    return overnight_ret, intraday_ret, overnight_ratio
