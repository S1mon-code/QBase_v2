import numpy as np

from indicators._utils import _ema_no_warmup as _ema


def tema(data: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average.

    Formula: TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

    Further reduces lag beyond DEMA by incorporating a third level of
    exponential smoothing.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    ema1 = _ema(data, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    return 3.0 * ema1 - 3.0 * ema2 + ema3
