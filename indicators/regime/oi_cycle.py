import numpy as np


def oi_cycle(oi: np.ndarray, period: int = 60) -> tuple:
    """OI cycle detection.

    OI tends to build up and wind down in cycles as market
    participants enter and exit.  This indicator estimates the
    current phase of the OI cycle and its amplitude.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    period : int
        Expected cycle length (lookback window).

    Returns
    -------
    cycle_phase : np.ndarray
        Phase of the OI cycle in [0, 1].
        0.0 = trough (OI at local min), 0.5 = peak (OI at local max).
    cycle_amplitude : np.ndarray
        Normalised amplitude: (oi - min) / (max - min) over
        the lookback.
    """
    n = len(oi)
    cycle_phase = np.full(n, np.nan)
    cycle_amplitude = np.full(n, np.nan)

    if n < period:
        return cycle_phase, cycle_amplitude

    for i in range(period - 1, n):
        window = oi[i - period + 1:i + 1]
        w_min = np.min(window)
        w_max = np.max(window)
        rng = w_max - w_min

        # Amplitude
        if rng > 0:
            cycle_amplitude[i] = (oi[i] - w_min) / rng
        else:
            cycle_amplitude[i] = 0.5

        # Phase: find position of min and max in window
        min_idx = np.argmin(window)
        max_idx = np.argmax(window)
        cur_idx = period - 1  # current bar is the last element

        # Determine phase based on whether we're closer to a
        # trough or peak and whether we're rising or falling
        if rng == 0:
            cycle_phase[i] = 0.5
            continue

        # Use normalised position as a proxy for phase
        # When amplitude is near 1.0 we're at a peak (phase~0.5)
        # When amplitude is near 0.0 we're at a trough (phase~0.0 or 1.0)
        amp = cycle_amplitude[i]

        # Determine direction (rising or falling)
        if cur_idx > 0:
            rising = window[-1] > window[-2]
        else:
            rising = True

        if rising:
            # Rising towards peak: phase 0.0 → 0.5
            cycle_phase[i] = amp * 0.5
        else:
            # Falling from peak: phase 0.5 → 1.0
            cycle_phase[i] = 0.5 + (1.0 - amp) * 0.5

    return cycle_phase, cycle_amplitude
