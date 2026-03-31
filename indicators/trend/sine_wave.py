import numpy as np


def ehlers_sine_wave(closes: np.ndarray, alpha: float = 0.07) -> tuple:
    """Ehlers Sine Wave indicator.

    Decomposes the dominant cycle using Hilbert Transform concepts
    and generates sine and lead_sine lines. Their crossover signals
    trend mode (parallel) vs cycle mode (crossing).

    Returns (sine, lead_sine).
    """
    if closes.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy()
    n = closes.size
    if n < 10:
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr.copy()

    # Smooth price
    smooth = np.full(n, 0.0)
    for i in range(3, n):
        smooth[i] = (
            closes[i] + 2.0 * closes[i - 1] + 2.0 * closes[i - 2] + closes[i - 3]
        ) / 6.0
    for i in range(3):
        smooth[i] = closes[i]

    # Cycle extraction using 2-pole IIR
    cycle = np.zeros(n)
    c1 = (1.0 - 0.5 * alpha) ** 2
    c2 = 2.0 * (1.0 - alpha)
    c3 = (1.0 - alpha) ** 2

    for i in range(6, n):
        cycle[i] = (
            c1 * (smooth[i] - 2.0 * smooth[i - 1] + smooth[i - 2])
            + c2 * cycle[i - 1]
            - c3 * cycle[i - 2]
        )

    # Compute instantaneous period using simple zero-crossing
    inst_period = np.full(n, 20.0)
    dc_phase = np.zeros(n)

    for i in range(1, n):
        # Accumulate phase using cycle and its quadrature
        if abs(cycle[i]) > 1e-10:
            # Simple phase estimation
            dc_phase[i] = np.arctan(cycle[i - 1] / cycle[i]) if abs(cycle[i]) > 1e-10 else dc_phase[i - 1]
        else:
            dc_phase[i] = dc_phase[i - 1]

    # Sine wave from dominant cycle
    sine = np.full(n, np.nan)
    lead_sine = np.full(n, np.nan)

    # Use a simpler approach: compute phase from analytic signal
    for i in range(7, n):
        # Quadrature (Hilbert approximation via difference)
        q = cycle[i] - cycle[i - 1]
        ip = cycle[i]

        if abs(ip) > 1e-10 or abs(q) > 1e-10:
            phase = np.arctan2(q, ip)
        else:
            phase = 0.0

        sine[i] = np.sin(phase)
        lead_sine[i] = np.sin(phase + np.pi / 4.0)  # 45 degree lead

    return sine, lead_sine
