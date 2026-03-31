import numpy as np


def normalized_volume(
    volumes: np.ndarray,
    period: int = 20,
) -> tuple:
    """Normalized Volume = Volume / SMA(Volume).

    Values > 1 mean above-average volume, < 1 mean below-average.
    Also computes rolling percentile rank of volume.
    Returns (norm_vol, vol_percentile).
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    volumes = volumes.astype(np.float64)

    norm_vol = np.full(n, np.nan)
    vol_percentile = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = volumes[i - period + 1 : i + 1]
        sma = np.mean(window)
        if sma > 0:
            norm_vol[i] = volumes[i] / sma

        # Percentile rank: what fraction of the window is <= current volume
        vol_percentile[i] = np.sum(window <= volumes[i]) / period * 100.0

    return norm_vol, vol_percentile
