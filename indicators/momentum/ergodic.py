import numpy as np

from indicators._utils import _ema, _ema_skip_nan


def ergodic(
    closes: np.ndarray,
    short_period: int = 5,
    long_period: int = 20,
    signal_period: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Ergodic Oscillator (William Blau).

    Double-smoothed momentum oscillator based on the True Strength Index.
    Ergodic = 100 * EMA(EMA(momentum, long), short) / EMA(EMA(|momentum|, long), short)
    Signal = EMA(Ergodic, signal_period)

    Returns (ergodic_line, signal_line).
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

    # Double smooth: EMA(long) then EMA(short)
    ema1_mom = _ema(momentum, long_period)
    ema2_mom = _ema_skip_nan(ema1_mom, short_period)

    ema1_abs = _ema(abs_momentum, long_period)
    ema2_abs = _ema_skip_nan(ema1_abs, short_period)

    # Ergodic line (momentum array is n-1, indices shifted by 1)
    ergo_raw = np.full(n - 1, np.nan)
    valid = (~np.isnan(ema2_mom)) & (~np.isnan(ema2_abs)) & (ema2_abs != 0)
    ergo_raw[valid] = 100.0 * ema2_mom[valid] / ema2_abs[valid]

    ergo_line = np.full(n, np.nan)
    ergo_line[1:] = ergo_raw

    # Signal line
    sig_raw = _ema_skip_nan(ergo_raw, signal_period)
    signal_line = np.full(n, np.nan)
    signal_line[1:] = sig_raw

    return ergo_line, signal_line
