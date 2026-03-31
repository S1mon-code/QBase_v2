import numpy as np


def intraday_intensity(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> tuple:
    """Intraday Intensity Index.

    Measures where the close falls within the high-low range, weighted by volume.
    II = (2*Close - High - Low) / (High - Low) * Volume
    Returns (ii, ii_signal) where ii_signal is the SMA of the cumulative II ratio.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    hl_range = highs - lows
    raw_ii = np.zeros(n, dtype=np.float64)
    mask = hl_range > 0
    raw_ii[mask] = (2.0 * closes[mask] - highs[mask] - lows[mask]) / hl_range[mask] * volumes[mask]

    # Cumulative II ratio: sum(II) / sum(Volume) over period
    ii = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sum = np.sum(volumes[i - period + 1 : i + 1])
        if vol_sum > 0:
            ii[i] = np.sum(raw_ii[i - period + 1 : i + 1]) / vol_sum

    # Signal line: SMA of ii
    sig_period = max(period // 2, 3)
    ii_signal = np.full(n, np.nan)
    for i in range(n):
        if i < sig_period - 1:
            continue
        window = ii[i - sig_period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == sig_period:
            ii_signal[i] = np.mean(valid)

    return ii, ii_signal
