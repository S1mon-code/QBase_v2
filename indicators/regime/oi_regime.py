import numpy as np


def oi_regime(closes: np.ndarray, oi: np.ndarray, volumes: np.ndarray,
              period: int = 60) -> tuple:
    """Comprehensive OI-based regime detection (Wyckoff-like).

    Classifies the market into four phases based on the interaction
    of price trend, OI trend, and volume trend:

        0 = accumulation  (price flat/down, OI rising, volume low)
        1 = markup         (price up, OI up, volume rising)
        2 = distribution   (price flat/up, OI falling, volume high)
        3 = markdown        (price down, OI down, volume rising)

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volume.
    period : int
        Lookback for trend estimation.

    Returns
    -------
    regime : np.ndarray (int)
        Regime label 0-3.  -1 during warmup.
    regime_name_id : np.ndarray (int)
        Same as regime (kept for clarity: 0=accum, 1=markup,
        2=distrib, 3=markdown).
    """
    n = len(closes)
    regime = np.full(n, -1, dtype=int)
    regime_name_id = np.full(n, -1, dtype=int)

    if n <= period:
        return regime, regime_name_id

    # Compute trends via linear regression slope sign
    def _slope(arr, start, end):
        y = arr[start:end]
        x = np.arange(len(y), dtype=float)
        xm = np.mean(x)
        ym = np.mean(y)
        denom = np.sum((x - xm) ** 2)
        if denom == 0:
            return 0.0
        return np.sum((x - xm) * (y - ym)) / denom

    half = period // 2

    for i in range(period, n):
        price_slope = _slope(closes, i - half, i + 1)
        oi_slope = _slope(oi, i - half, i + 1)

        # Volume: compare recent half to prior half
        vol_recent = np.mean(volumes[i - half:i + 1])
        vol_prior = np.mean(volumes[i - period:i - half])
        vol_rising = vol_recent > vol_prior

        # Normalise slopes for comparison
        price_up = price_slope > 0
        oi_up = oi_slope > 0

        if not price_up and oi_up and not vol_rising:
            r = 0  # accumulation
        elif price_up and oi_up:
            r = 1  # markup
        elif price_up and not oi_up:
            r = 2  # distribution
        else:
            r = 3  # markdown

        regime[i] = r
        regime_name_id[i] = r

    return regime, regime_name_id
