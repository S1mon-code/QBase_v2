import numpy as np


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average with NaN warmup."""
    n = data.size
    out = np.full(n, np.nan)
    if n < period:
        return out
    cs = np.cumsum(data)
    out[period - 1] = cs[period - 1] / period
    out[period:] = (cs[period:] - cs[:-period]) / period
    return out


def acceleration_bands(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
    width: float = 0.04,
) -> tuple:
    """Headley's Acceleration Bands.

    Upper = SMA(high * (1 + width * (high - low) / (high + low)))
    Lower = SMA(low  * (1 - width * (high - low) / (high + low)))
    Mid   = SMA(close)

    Returns (upper, mid, lower) arrays.
    """
    if closes.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()
    n = closes.size
    if n < period:
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr.copy(), nan_arr.copy()

    hl_sum = highs + lows
    # Avoid division by zero
    safe_sum = np.where(np.abs(hl_sum) < 1e-12, 1e-12, hl_sum)
    factor = width * (highs - lows) / safe_sum

    upper_raw = highs * (1.0 + factor)
    lower_raw = lows * (1.0 - factor)

    upper = _sma(upper_raw, period)
    mid = _sma(closes, period)
    lower = _sma(lower_raw, period)

    return upper, mid, lower
