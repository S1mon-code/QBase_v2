"""Volume clock: volume-weighted time progression.

Converts calendar time into "volume time" by measuring how fast
volume is accumulating relative to its average pace.
"""

import numpy as np


def volume_clock(
    volumes: np.ndarray,
    target_volume: float | None = None,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Volume-weighted time progression.

    Parameters
    ----------
    volumes       : volume per bar.
    target_volume : expected volume per bar.  If None, uses the rolling
                    mean over *period* as the target.
    period        : lookback for rolling average volume.

    Returns
    -------
    (clock_speed, acceleration)
        clock_speed  – current volume / expected volume.
                       > 1 means faster than average (high activity).
                       < 1 means slower (low activity).
        acceleration – change in clock_speed (first derivative).
                       Positive = activity is increasing.
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    clock_speed = np.full(n, np.nan)
    acceleration = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = volumes[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            continue

        if target_volume is not None and target_volume > 0:
            avg_vol = target_volume
        else:
            avg_vol = np.mean(valid)

        if avg_vol > 0 and not np.isnan(volumes[i]):
            clock_speed[i] = volumes[i] / avg_vol

    # Acceleration: change in clock speed
    for i in range(1, n):
        if not np.isnan(clock_speed[i]) and not np.isnan(clock_speed[i - 1]):
            acceleration[i] = clock_speed[i] - clock_speed[i - 1]

    return clock_speed, acceleration
