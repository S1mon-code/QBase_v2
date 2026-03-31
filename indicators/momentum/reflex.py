import numpy as np


def reflex(closes: np.ndarray, period: int = 20) -> np.ndarray:
    """Ehlers Reflex indicator — measures trend reversal.

    Compares the current value of a super-smoother filter to its value
    'period' bars ago, normalized by RMS of the difference. Positive
    values indicate upward reversal, negative values downward.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    warmup = period + 10
    if n <= warmup:
        return np.full(n, np.nan)

    # Super Smoother (2-pole Butterworth)
    a1 = np.exp(-np.sqrt(2.0) * np.pi / period)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2.0) * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3

    filt = np.full(n, np.nan)
    filt[0] = closes[0]
    filt[1] = closes[1]
    for i in range(2, n):
        filt[i] = c1 * (closes[i] + closes[i - 1]) / 2.0 + c2 * filt[i - 1] + c3 * filt[i - 2]

    # Reflex
    out = np.full(n, np.nan)
    ms = np.zeros(n)

    for i in range(period, n):
        slope = (filt[i - period] - filt[i]) / period

        # Sum of differences along the slope line
        _sum = 0.0
        for j in range(1, period + 1):
            _sum += filt[i] + j * slope - filt[i - j]

        _sum /= period

        # RMS (exponential moving)
        ms[i] = 0.04 * _sum * _sum + 0.96 * ms[i - 1]
        rms = np.sqrt(ms[i])

        if rms > 1e-12:
            out[i] = _sum / rms
        else:
            out[i] = 0.0

    return out
