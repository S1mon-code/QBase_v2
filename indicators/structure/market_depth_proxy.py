import numpy as np


def depth_proxy(highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray,
                period: int = 20) -> tuple:
    """Market depth proxy from high-low range and volume.

    Depth = Volume / Range. High depth means lots of volume within a
    narrow range, implying thick order book / high liquidity.

    Parameters
    ----------
    highs : np.ndarray
        High prices.
    lows : np.ndarray
        Low prices.
    volumes : np.ndarray
        Trading volumes.
    period : int
        Rolling period for z-score computation.

    Returns
    -------
    depth : np.ndarray
        Raw depth proxy (volume / range).
    depth_zscore : np.ndarray
        Z-score of depth over rolling window.
    """
    n = len(highs)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    depth = np.full(n, np.nan)
    depth_z = np.full(n, np.nan)

    for i in range(n):
        rng = highs[i] - lows[i]
        if (rng > 1e-9 and np.isfinite(volumes[i]) and volumes[i] > 0
                and np.isfinite(highs[i]) and np.isfinite(lows[i])):
            depth[i] = volumes[i] / rng

    # Rolling z-score
    for i in range(period, n):
        window = depth[i - period:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < period // 2:
            continue
        mu = np.mean(valid)
        std = np.std(valid)
        if std > 1e-9 and np.isfinite(depth[i]):
            depth_z[i] = (depth[i] - mu) / std

    return depth, depth_z
