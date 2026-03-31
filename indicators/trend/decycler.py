import numpy as np


def decycler(closes: np.ndarray, period: int = 60) -> np.ndarray:
    """Ehlers Simple Decycler — removes cycle component, keeps trend.

    Applies a high-pass filter and subtracts it from price to remove
    cyclic components shorter than 'period' bars. The result is a
    smooth trend line that follows price with minimal lag.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n < 3:
        return np.full(n, np.nan)

    # 2-pole high-pass filter coefficients
    alpha1 = (np.cos(0.707 * 2.0 * np.pi / period) + np.sin(0.707 * 2.0 * np.pi / period) - 1.0) / np.cos(0.707 * 2.0 * np.pi / period)

    hp = np.zeros(n)
    hp[0] = 0.0
    hp[1] = 0.0

    c1 = (1.0 - alpha1 / 2.0) * (1.0 - alpha1 / 2.0)
    c2 = 2.0 * (1.0 - alpha1)
    c3 = (1.0 - alpha1) * (1.0 - alpha1)

    for i in range(2, n):
        hp[i] = c1 * (closes[i] - 2.0 * closes[i - 1] + closes[i - 2]) + c2 * hp[i - 1] - c3 * hp[i - 2]

    # Decycled = price - high_pass (i.e., the low-frequency trend)
    out = np.full(n, np.nan)
    warmup = min(period, n)
    for i in range(warmup, n):
        out[i] = closes[i] - hp[i]

    return out
