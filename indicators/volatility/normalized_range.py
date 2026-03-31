import numpy as np


def normalized_range(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
) -> tuple:
    """Normalized Range: (High - Low) / Close, smoothed over period.

    Lower values indicate price compression (potential breakout setup).
    Returns (nr, nr_zscore).
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)

    # Raw normalized range per bar
    raw_nr = np.full(n, np.nan)
    mask = closes > 0
    raw_nr[mask] = (highs[mask] - lows[mask]) / closes[mask]

    # Smoothed NR: SMA over period
    nr = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = raw_nr[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == period:
            nr[i] = np.mean(valid)

    # Z-score over a longer lookback (2x period)
    z_lookback = period * 2
    nr_zscore = np.full(n, np.nan)
    for i in range(z_lookback - 1, n):
        window = nr[i - z_lookback + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= period:
            mu = np.mean(valid)
            sigma = np.std(valid, ddof=1)
            if sigma > 0 and not np.isnan(nr[i]):
                nr_zscore[i] = (nr[i] - mu) / sigma

    return nr, nr_zscore
