import numpy as np


def ou_speed(
    data: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Ornstein-Uhlenbeck mean reversion speed (half-life) estimator.

    Fits an AR(1) model on log prices within each rolling window:
        x_t - x_{t-1} = theta * (mu - x_{t-1}) + noise
    Speed theta is estimated via OLS regression of dx on x_{t-1}.

    Returns (speed, half_life). High speed = fast mean reversion.
    half_life = ln(2) / speed (in bars). NaN when speed <= 0 (trending).
    """
    n = len(data)
    speed = np.full(n, np.nan, dtype=np.float64)
    half_life = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return speed, half_life

    safe = np.maximum(data, 1e-9)
    log_p = np.log(safe)

    if n < period + 1:
        return speed, half_life

    for i in range(period, n):
        window = log_p[i - period : i + 1]
        y = np.diff(window)  # dx = x_t - x_{t-1}, length period
        x = window[:-1]      # x_{t-1}, length period

        # OLS: y = a + b * x + eps
        # b < 0 means mean reverting, speed = -b
        x_dm = x - np.mean(x)
        y_dm = y - np.mean(y)

        denom = np.dot(x_dm, x_dm)
        if denom < 1e-14:
            continue

        b = np.dot(x_dm, y_dm) / denom
        theta = -b  # mean reversion speed

        speed[i] = theta

        if theta > 1e-10:
            half_life[i] = np.log(2.0) / theta
        # else: trending or random walk, half_life stays NaN

    return speed, half_life
