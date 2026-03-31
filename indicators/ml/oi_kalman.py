import numpy as np


def oi_kalman_trend(oi: np.ndarray, process_noise: float = 0.01,
                    measurement_noise: float = 1.0) -> tuple:
    """Kalman filter on OI to extract smooth trend and momentum.

    Uses a constant-velocity Kalman filter where the state is
    [oi_level, oi_velocity].  This provides a smooth OI trend
    and its first derivative (momentum) while filtering noise.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    process_noise : float
        Process noise covariance scaling (Q). Higher = more
        responsive, lower = smoother.
    measurement_noise : float
        Measurement noise variance (R). Higher = trust model more.

    Returns
    -------
    oi_trend : np.ndarray
        Kalman-filtered OI level (smooth trend).
    oi_momentum : np.ndarray
        Kalman-estimated OI velocity (first derivative).
    prediction_error : np.ndarray
        Innovation (measurement - prediction). Large errors
        indicate surprising OI changes.
    """
    n = len(oi)
    oi_trend = np.full(n, np.nan)
    oi_momentum = np.full(n, np.nan)
    prediction_error = np.full(n, np.nan)

    if n < 2:
        return oi_trend, oi_momentum, prediction_error

    # State: [level, velocity]
    # Transition: level' = level + velocity, velocity' = velocity
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])  # observe level only

    Q = process_noise * np.array([[0.25, 0.5],
                                   [0.5, 1.0]])
    R = np.array([[measurement_noise]])

    # Initialise state
    x = np.array([oi[0], 0.0])  # [level, velocity=0]
    P = np.eye(2) * 1000.0  # high initial uncertainty

    for i in range(n):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Innovation
        z = oi[i]
        y = z - (H @ x_pred)[0]
        prediction_error[i] = y

        # Kalman gain
        S = (H @ P_pred @ H.T + R)[0, 0]
        if S > 0:
            K = (P_pred @ H.T) / S
        else:
            K = np.zeros((2, 1))

        # Update
        x = x_pred + (K * y).flatten()
        P = (np.eye(2) - K @ H) @ P_pred

        oi_trend[i] = x[0]
        oi_momentum[i] = x[1]

    return oi_trend, oi_momentum, prediction_error
