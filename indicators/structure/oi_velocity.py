import numpy as np


def oi_velocity(oi: np.ndarray, period: int = 5) -> tuple:
    """Speed and acceleration of OI change.

    First derivative (velocity) measures the rate of OI change.
    Second derivative (acceleration) measures how fast the rate changes.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    period : int
        Smoothing period for derivative estimation.

    Returns
    -------
    velocity : np.ndarray
        Smoothed first derivative of OI (rate of change).
    acceleration : np.ndarray
        Smoothed second derivative of OI (change of rate).
    """
    n = len(oi)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    velocity = np.full(n, np.nan)
    acceleration = np.full(n, np.nan)

    # First derivative: smoothed OI change
    for i in range(period, n):
        window = oi[i - period:i + 1]
        valid = np.isfinite(window)
        if np.sum(valid) < period:
            continue
        # Linear regression slope as smoothed derivative
        x = np.arange(period + 1, dtype=float)
        y = window.copy()
        if not np.all(valid):
            continue
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx > 0:
            velocity[i] = np.sum((x - x_mean) * (y - y_mean)) / ss_xx

    # Second derivative: change in velocity
    for i in range(period + period, n):
        v_window = velocity[i - period:i + 1]
        valid = np.isfinite(v_window)
        if np.sum(valid) < period:
            continue
        if not np.all(valid):
            continue
        x = np.arange(period + 1, dtype=float)
        y = v_window.copy()
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx > 0:
            acceleration[i] = np.sum((x - x_mean) * (y - y_mean)) / ss_xx

    return velocity, acceleration
