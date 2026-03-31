import numpy as np

from indicators._utils import _ema


def _ema_zero_fill(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with NaN warmup."""
    n = data.size
    out = np.full(n, np.nan)
    if n < period:
        return out
    alpha = 2.0 / (period + 1)
    # Seed with SMA
    out[period - 1] = np.mean(data[:period])
    for i in range(period, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def trix(closes: np.ndarray, period: int = 15) -> tuple:
    """TRIX — Triple EXponential moving average rate of change.

    Applies EMA three times, then computes the 1-bar rate of change
    of the result. Filters out short-term noise very effectively.

    Returns (trix_line, signal) where signal is a 9-period EMA of trix.
    """
    if closes.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy()
    n = closes.size
    warmup = 3 * period + 1
    if n <= warmup:
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr.copy()

    # Triple EMA
    ema1 = _ema(closes, period)

    # For ema2 and ema3, replace NaN with 0 before feeding (then re-NaN)
    ema1_clean = np.where(np.isnan(ema1), 0.0, ema1)
    ema2 = _ema(ema1_clean, period)
    ema2[:2 * period - 2] = np.nan

    ema2_clean = np.where(np.isnan(ema2), 0.0, ema2)
    ema3 = _ema(ema2_clean, period)
    ema3[:3 * period - 3] = np.nan

    # Rate of change of triple EMA (percentage)
    trix_line = np.full(n, np.nan)
    for i in range(3 * period - 2, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i - 1]) and abs(ema3[i - 1]) > 1e-12:
            trix_line[i] = (ema3[i] - ema3[i - 1]) / ema3[i - 1] * 10000.0
        else:
            trix_line[i] = 0.0

    # Signal line: 9-period EMA of trix
    signal_period = 9
    trix_clean = np.where(np.isnan(trix_line), 0.0, trix_line)
    signal = _ema(trix_clean, signal_period)
    # NaN out warmup
    signal[:3 * period - 2 + signal_period - 1] = np.nan

    return trix_line, signal
