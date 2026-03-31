import numpy as np

from indicators._utils import _ema_no_warmup as _ema


def dema(data: np.ndarray, period: int) -> np.ndarray:
    """Double Exponential Moving Average.

    Formula: DEMA = 2 * EMA(data, period) - EMA(EMA(data, period), period)

    Reduces lag compared to a standard EMA by subtracting the smoothed EMA
    from twice the original EMA.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    ema1 = _ema(data, period)
    ema2 = _ema(ema1, period)
    return 2.0 * ema1 - ema2
