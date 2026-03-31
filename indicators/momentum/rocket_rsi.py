import numpy as np


def rocket_rsi(
    closes: np.ndarray,
    rsi_period: int = 10,
    rocket_period: int = 8,
) -> np.ndarray:
    """Roger Altman's Rocket RSI — Fisher Transform of RSI.

    Computes RSI, normalizes it to [-1, 1], then applies the Fisher
    Transform to create a Gaussian-distributed oscillator. Extreme
    values (> 2 or < -2) indicate overbought/oversold conditions.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    warmup = rsi_period + rocket_period
    if n <= warmup:
        return np.full(n, np.nan)

    # RSI calculation
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rsi_arr = np.full(n, np.nan)

    # Seed with SMA
    avg_gain[rsi_period] = np.mean(gains[1 : rsi_period + 1])
    avg_loss[rsi_period] = np.mean(losses[1 : rsi_period + 1])

    for i in range(rsi_period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gains[i]) / rsi_period
        avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + losses[i]) / rsi_period

    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-12:
            rsi_arr[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_arr[i] = 100.0 - 100.0 / (1.0 + rs)

    # Normalize RSI to [-1, 1] range using rolling min/max over rocket_period
    out = np.full(n, np.nan)
    for i in range(warmup, n):
        window = rsi_arr[i - rocket_period + 1 : i + 1]
        lo = np.nanmin(window)
        hi = np.nanmax(window)
        rng = hi - lo
        if rng < 1e-12:
            normed = 0.0
        else:
            normed = 2.0 * (rsi_arr[i] - lo) / rng - 1.0

        # Clamp to avoid log(0)
        normed = max(-0.999, min(0.999, normed))

        # Fisher Transform
        out[i] = 0.5 * np.log((1.0 + normed) / (1.0 - normed))

    return out
