import numpy as np


def cog(closes: np.ndarray, period: int = 10) -> np.ndarray:
    """Ehlers Center of Gravity oscillator.

    A leading indicator based on FIR (Finite Impulse Response) filter.
    COG = -Sum(close[i-j] * (j+1), j=0..period-1) / Sum(close[i-j], j=0..period-1)

    Returns values that oscillate around zero. Zero-line crossovers
    can signal trend changes.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n < period:
        return np.full(n, np.nan)

    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        weights = np.arange(period, 0, -1, dtype=float)  # period, period-1, ..., 1
        den = np.sum(window)
        if abs(den) > 1e-12:
            out[i] = -np.sum(window * weights) / den
        else:
            out[i] = 0.0

    return out
