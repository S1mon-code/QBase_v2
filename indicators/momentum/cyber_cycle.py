import numpy as np


def cyber_cycle(closes: np.ndarray, alpha: float = 0.07) -> tuple:
    """Ehlers Cyber Cycle — IIR bandpass filter extracting dominant cycle.

    Uses a 2-pole high-pass filter followed by smoothing to isolate
    the dominant cycle component from price data.

    Returns (cycle, signal) where signal is a 1-bar lag trigger line.
    """
    if closes.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy()
    n = closes.size
    if n < 7:
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr.copy()

    # Smooth price with a 4-bar WMA first
    smooth = np.full(n, np.nan)
    for i in range(3, n):
        smooth[i] = (
            closes[i] + 2.0 * closes[i - 1] + 2.0 * closes[i - 2] + closes[i - 3]
        ) / 6.0

    # Cyber Cycle: 2-pole IIR bandpass filter
    cycle = np.full(n, np.nan)
    # Initialize first few values
    cycle[0] = 0.0
    cycle[1] = 0.0
    cycle[2] = 0.0
    cycle[3] = 0.0
    cycle[4] = 0.0
    cycle[5] = 0.0

    c1 = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha)
    c2 = 2.0 * (1.0 - alpha)
    c3 = (1.0 - alpha) * (1.0 - alpha)

    for i in range(6, n):
        if np.isnan(smooth[i]) or np.isnan(smooth[i - 2]):
            cycle[i] = 0.0
        else:
            cycle[i] = (
                c1 * (smooth[i] - 2.0 * smooth[i - 1] + smooth[i - 2])
                + c2 * cycle[i - 1]
                - c3 * cycle[i - 2]
            )

    # Signal line: 1-bar advance trigger
    signal = np.full(n, np.nan)
    for i in range(7, n):
        signal[i] = cycle[i - 1]

    # Set warmup to NaN
    cycle[:6] = np.nan
    signal[:7] = np.nan

    return cycle, signal
