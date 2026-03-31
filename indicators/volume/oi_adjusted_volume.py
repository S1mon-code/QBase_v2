import numpy as np


def oi_adjusted_volume(volumes: np.ndarray, oi: np.ndarray,
                       period: int = 20) -> tuple:
    """Volume adjusted by OI level.

    Normalises volume by total open interest to measure turnover
    relative to outstanding positions.  High adjusted volume means
    active churning; low means positions are static.

    Parameters
    ----------
    volumes : np.ndarray
        Trading volume.
    oi : np.ndarray
        Open interest.
    period : int
        Lookback for z-score calculation.

    Returns
    -------
    adj_volume : np.ndarray
        Volume / OI ratio (raw).
    adj_volume_zscore : np.ndarray
        Z-score of adj_volume over *period*.
    """
    n = len(volumes)
    adj_volume = np.full(n, np.nan)
    adj_volume_zscore = np.full(n, np.nan)

    for i in range(n):
        if oi[i] > 0:
            adj_volume[i] = volumes[i] / oi[i]
        else:
            adj_volume[i] = 0.0

    if n < period:
        return adj_volume, adj_volume_zscore

    for i in range(period - 1, n):
        window = adj_volume[i - period + 1:i + 1]
        m = np.mean(window)
        s = np.std(window, ddof=1)
        if s > 0:
            adj_volume_zscore[i] = (adj_volume[i] - m) / s
        else:
            adj_volume_zscore[i] = 0.0

    return adj_volume, adj_volume_zscore
