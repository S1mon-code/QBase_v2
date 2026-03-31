import numpy as np


def oi_relative_strength(oi: np.ndarray, volumes: np.ndarray,
                         period: int = 20) -> tuple:
    """OI strength relative to volume.

    Measures whether positions are accumulating faster than trading
    activity.  A rising oi_rs means OI is growing relative to volume
    (positions being built), falling means positions are being
    unwound relative to activity.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volume.
    period : int
        Lookback for trend calculation.

    Returns
    -------
    oi_rs : np.ndarray
        OI-to-volume ratio (smoothed).
    oi_rs_trend : np.ndarray
        Rate of change of oi_rs over *period* bars.
    """
    n = len(oi)
    oi_rs = np.full(n, np.nan)
    oi_rs_trend = np.full(n, np.nan)

    if n < period:
        return oi_rs, oi_rs_trend

    # Rolling OI / Volume ratio (smoothed with SMA)
    for i in range(period - 1, n):
        oi_win = oi[i - period + 1:i + 1]
        vol_win = volumes[i - period + 1:i + 1]
        sum_vol = np.sum(vol_win)
        if sum_vol > 0:
            oi_rs[i] = np.mean(oi_win) / (sum_vol / period)
        else:
            oi_rs[i] = 0.0

    # Trend: rate of change of oi_rs
    for i in range(2 * period - 2, n):
        prev = oi_rs[i - period + 1]
        if not np.isnan(prev) and prev > 0:
            oi_rs_trend[i] = (oi_rs[i] - prev) / prev
        elif not np.isnan(prev):
            oi_rs_trend[i] = 0.0

    return oi_rs, oi_rs_trend
