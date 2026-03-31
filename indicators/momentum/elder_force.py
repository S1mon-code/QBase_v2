import numpy as np

from indicators._utils import _ema


def elder_force_index(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 13,
) -> np.ndarray:
    """Elder Force Index.

    Raw Force = (Close - Prior Close) * Volume
    EFI = EMA(Raw Force, period)

    Combines price change and volume to measure the force behind moves.
    No fixed range.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n <= period:
        return np.full(n, np.nan)

    # 1-period force index
    raw_force = np.diff(closes) * volumes[1:]

    # EMA smoothing
    ema_force = _ema(raw_force, period)

    # Map back to full-size array (index 0 is NaN)
    result = np.full(n, np.nan)
    result[1:] = ema_force

    return result
