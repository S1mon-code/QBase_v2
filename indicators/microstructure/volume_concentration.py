"""Volume concentration — what fraction of total volume comes from top bars?

High concentration indicates institutional activity or event-driven
trading where volume clusters in a few bars.
"""

import numpy as np


def volume_concentration(
    volumes: np.ndarray,
    period: int = 20,
    top_pct: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Volume concentration: share of total volume from top bars.

    Parameters
    ----------
    volumes : array of trading volumes.
    period  : rolling window.
    top_pct : fraction of bars considered "top" (default 0.2 = top 20%).

    Returns
    -------
    (concentration, is_concentrated)
        concentration    – fraction of total volume from top_pct bars.
                           Range [top_pct, 1].  Higher = more concentrated.
        is_concentrated  – 1.0 if concentration > 0.8 (Pareto-like), else 0.0.
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    concentration = np.full(n, np.nan)
    is_conc = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = volumes[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 3:
            continue

        total = np.sum(valid)
        if total <= 0:
            continue

        k = max(1, int(len(valid) * top_pct))
        sorted_v = np.sort(valid)[::-1]  # descending
        top_sum = np.sum(sorted_v[:k])

        concentration[i] = top_sum / total
        is_conc[i] = 1.0 if concentration[i] > 0.8 else 0.0

    return concentration, is_conc
