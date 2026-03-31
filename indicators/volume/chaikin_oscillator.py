import numpy as np

from indicators._utils import _ema


def chaikin_oscillator(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    fast: int = 3,
    slow: int = 10,
) -> np.ndarray:
    """Chaikin Oscillator = EMA(fast) of A/D Line - EMA(slow) of A/D Line.

    Measures momentum of the Accumulation/Distribution line.
    Positive values indicate accumulation momentum, negative indicate distribution.
    Returns chaikin_osc array.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    # A/D Line
    hl_range = highs - lows
    clv = np.zeros(n, dtype=np.float64)
    mask = hl_range > 0
    clv[mask] = ((closes[mask] - lows[mask]) - (highs[mask] - closes[mask])) / hl_range[mask]
    ad = np.cumsum(clv * volumes)

    # EMA of A/D
    ema_fast = _ema(ad, fast)
    ema_slow = _ema(ad, slow)

    chaikin_osc = np.full(n, np.nan)
    valid = (~np.isnan(ema_fast)) & (~np.isnan(ema_slow))
    chaikin_osc[valid] = ema_fast[valid] - ema_slow[valid]

    return chaikin_osc
