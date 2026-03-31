import numpy as np


def price_acceleration(
    closes: np.ndarray,
    period: int = 14,
) -> tuple:
    """Second derivative of price (acceleration) and third derivative (jerk).

    Acceleration > 0: price momentum increasing. < 0: momentum fading.
    Jerk measures the rate of change of acceleration.
    Returns (acceleration, jerk, is_accelerating).
    is_accelerating: 1.0 when acceleration > 0, 0.0 otherwise, NaN during warmup.
    """
    n = len(closes)
    if n < 3:
        emp = np.full(n, np.nan)
        return emp.copy(), emp.copy(), emp.copy()

    closes = closes.astype(np.float64)

    # First derivative: rate of change (velocity)
    velocity = np.full(n, np.nan)
    for i in range(period, n):
        velocity[i] = (closes[i] - closes[i - period]) / closes[i - period]

    # Second derivative: acceleration (change of velocity)
    acceleration = np.full(n, np.nan)
    for i in range(period * 2, n):
        if not np.isnan(velocity[i]) and not np.isnan(velocity[i - period]):
            acceleration[i] = velocity[i] - velocity[i - period]

    # Third derivative: jerk (change of acceleration)
    jerk = np.full(n, np.nan)
    for i in range(period * 3, n):
        if not np.isnan(acceleration[i]) and not np.isnan(acceleration[i - period]):
            jerk[i] = acceleration[i] - acceleration[i - period]

    # Boolean: is accelerating
    is_accelerating = np.full(n, np.nan)
    valid = ~np.isnan(acceleration)
    is_accelerating[valid & (acceleration > 0)] = 1.0
    is_accelerating[valid & (acceleration <= 0)] = 0.0

    return acceleration, jerk, is_accelerating
