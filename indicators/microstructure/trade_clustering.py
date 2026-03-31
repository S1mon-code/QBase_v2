"""Detect clustering of high-volume bars.

When high-volume bars cluster together, it signals sustained
institutional activity rather than random spikes.
"""

import numpy as np


def trade_clustering(
    volumes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Trade clustering: detect clusters of high-volume bars.

    Parameters
    ----------
    volumes : array of trading volumes.
    period  : rolling window.

    Returns
    -------
    (cluster_score, is_clustered)
        cluster_score – fraction of bars in the window that are "high volume"
                        (above 1.5x rolling mean).  Range [0, 1].
        is_clustered  – 1.0 if cluster_score > 0.3, else 0.0.
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    cluster_score = np.full(n, np.nan)
    is_clustered = np.full(n, np.nan)

    # Compute rolling mean volume for threshold
    vol_mean = np.full(n, np.nan)
    lookback = max(period * 3, 60)
    for i in range(lookback - 1, n):
        window = volumes[i - lookback + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            vol_mean[i] = np.mean(valid)

    for i in range(max(period, lookback) - 1, n):
        if np.isnan(vol_mean[i]) or vol_mean[i] <= 0:
            continue
        threshold = vol_mean[i] * 1.5
        window = volumes[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 3:
            continue

        high_count = np.sum(valid > threshold)
        cluster_score[i] = high_count / len(valid)
        is_clustered[i] = 1.0 if cluster_score[i] > 0.3 else 0.0

    return cluster_score, is_clustered
