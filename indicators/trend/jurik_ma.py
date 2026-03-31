import numpy as np


def jma(
    closes: np.ndarray,
    period: int = 7,
    phase: float = 0.0,
    power: float = 2.0,
) -> np.ndarray:
    """Jurik Moving Average approximation — extremely low lag smoothing.

    An adaptive moving average with very low lag and smooth output.
    Phase controls the tradeoff between lag and overshoot:
      phase < 0: less overshoot, more lag
      phase > 0: less lag, more overshoot
    Power controls smoothness (higher = smoother).
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n < 2:
        return np.full(n, np.nan)

    # Phase to beta
    if phase < -100.0:
        phase_ratio = 0.5
    elif phase > 100.0:
        phase_ratio = 2.5
    else:
        phase_ratio = phase / 100.0 + 1.5

    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2.0)
    alpha = beta ** power

    e0 = np.zeros(n)
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    out = np.full(n, np.nan)

    e0[0] = closes[0]
    e1[0] = 0.0
    e2[0] = 0.0
    out[0] = closes[0]

    for i in range(1, n):
        e0[i] = (1.0 - alpha) * closes[i] + alpha * e0[i - 1]
        e1[i] = (closes[i] - e0[i]) * (1.0 - beta) + beta * e1[i - 1]
        e2[i] = (e0[i] + phase_ratio * e1[i] - out[i - 1]) * (1.0 - alpha) ** 2 + alpha ** 2 * e2[i - 1]
        out[i] = out[i - 1] + e2[i]

    # NaN warmup
    warmup = min(period, n)
    out[:warmup] = np.nan

    return out
