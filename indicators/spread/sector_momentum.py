"""Sector / group momentum diffusion index.

Takes a list of close-price arrays for a basket of assets and computes
aggregate momentum statistics: average momentum, breadth (fraction with
positive momentum), and a diffusion index.
"""

import numpy as np


def sector_momentum(
    closes_list: list[np.ndarray],
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sector/group momentum diffusion.

    Parameters
    ----------
    closes_list : list of close-price arrays (one per asset).
                  All arrays must have the same length.
    period      : lookback for rate-of-change calculation.

    Returns
    -------
    (avg_momentum, breadth, diffusion_index)
        avg_momentum    – average ROC across all assets.
        breadth         – fraction of assets with positive ROC (0-1).
        diffusion_index – 2 * breadth - 1 (ranges -1 to +1).
                          +1 = all assets rising, -1 = all falling.
    """
    if not closes_list:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    n = len(closes_list[0])
    k = len(closes_list)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    # Compute ROC for each asset
    rocs = np.full((k, n), np.nan)
    for j in range(k):
        c = closes_list[j]
        for i in range(period, n):
            prev = c[i - period]
            if prev != 0 and not np.isnan(prev):
                rocs[j, i] = (c[i] / prev - 1.0) * 100.0

    avg_momentum = np.full(n, np.nan)
    breadth = np.full(n, np.nan)
    diffusion = np.full(n, np.nan)

    for i in range(period, n):
        col = rocs[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        avg_momentum[i] = np.mean(valid)
        pos_frac = np.sum(valid > 0) / len(valid)
        breadth[i] = pos_frac
        diffusion[i] = 2.0 * pos_frac - 1.0

    return avg_momentum, breadth, diffusion
