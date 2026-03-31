import numpy as np

from indicators._utils import _ema, _ema_skip_nan


def tsi(
    closes: np.ndarray,
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """True Strength Index.

    TSI = 100 * EMA(EMA(momentum, long), short) / EMA(EMA(|momentum|, long), short)
    Signal = EMA(TSI, signal_period)

    Double-smoothed momentum oscillator. Typically ranges roughly -100 to 100.
    Returns (tsi_line, signal_line).
    """
    n = closes.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty
    nans = np.full(n, np.nan)
    if n <= long_period + short_period:
        return nans.copy(), nans.copy()

    momentum = np.diff(closes)
    abs_momentum = np.abs(momentum)

    # Double smooth: first EMA(long), then EMA(short) of result
    ema1_mom = _ema(momentum, long_period)
    ema2_mom = _ema_skip_nan(ema1_mom, short_period)

    ema1_abs = _ema(abs_momentum, long_period)
    ema2_abs = _ema_skip_nan(ema1_abs, short_period)

    # TSI (momentum array is size n-1, so result indices are shifted by 1)
    tsi_raw = np.full(n - 1, np.nan)
    valid = (~np.isnan(ema2_mom)) & (~np.isnan(ema2_abs)) & (ema2_abs != 0)
    tsi_raw[valid] = 100.0 * ema2_mom[valid] / ema2_abs[valid]

    # Put back into full-size array (index 0 is NaN because diff loses one element)
    tsi_line = np.full(n, np.nan)
    tsi_line[1:] = tsi_raw

    # Signal line
    sig = _ema_skip_nan(tsi_raw, signal_period)
    signal_line = np.full(n, np.nan)
    signal_line[1:] = sig

    return tsi_line, signal_line
