import numpy as np


def oi_breakout(oi: np.ndarray, period: int = 20,
                threshold: float = 2.0) -> tuple:
    """Detect OI breakout from range (sudden position buildup).

    An OI breakout occurs when the change in OI exceeds *threshold*
    standard deviations of its recent changes.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    period : int
        Lookback for range and std calculation.
    threshold : float
        Number of standard deviations to qualify as breakout.

    Returns
    -------
    is_breakout : np.ndarray (float)
        1.0 when OI breaks out, 0.0 otherwise.
    breakout_magnitude : np.ndarray
        Z-score of the OI change (magnitude of the breakout).
    breakout_direction : np.ndarray
        +1.0 for upward OI breakout (position buildup),
        -1.0 for downward (position unwind), 0.0 otherwise.
    """
    n = len(oi)
    is_breakout = np.zeros(n, dtype=float)
    breakout_magnitude = np.full(n, np.nan)
    breakout_direction = np.zeros(n, dtype=float)

    if n < period + 1:
        return is_breakout, breakout_magnitude, breakout_direction

    oi_change = np.full(n, np.nan)
    oi_change[1:] = np.diff(oi)

    for i in range(period + 1, n):
        window = oi_change[i - period:i]
        m = np.mean(window)
        s = np.std(window, ddof=1)

        if s > 0:
            z = (oi_change[i] - m) / s
            breakout_magnitude[i] = z

            if abs(z) >= threshold:
                is_breakout[i] = 1.0
                breakout_direction[i] = 1.0 if z > 0 else -1.0
        else:
            breakout_magnitude[i] = 0.0

    return is_breakout, breakout_magnitude, breakout_direction
