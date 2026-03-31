import numpy as np


def kalman_filter(
    closes: np.ndarray,
    process_noise: float = 0.01,
    measurement_noise: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kalman filter for trend estimation (level + slope model).

    Implements a two-state Kalman filter where the hidden state is
    [level, slope].  The slope is the instantaneous trend estimate and
    the prediction error (innovation) signals surprise moves.

    Parameters
    ----------
    closes : (N,) price series.
    process_noise : variance of the state transition noise (Q diagonal).
    measurement_noise : variance of the observation noise (R).

    Returns
    -------
    filtered_level : (N,) smoothed price level.
    filtered_slope : (N,) trend (slope) estimate; positive = up-trend.
    prediction_error : (N,) innovation = observed - predicted.
    """
    n = len(closes)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    level = np.full(n, np.nan, dtype=np.float64)
    slope = np.full(n, np.nan, dtype=np.float64)
    error = np.full(n, np.nan, dtype=np.float64)

    # State: x = [level, slope]
    # Transition: x_{t+1} = F @ x_t + noise
    # Observation: y_t = H @ x_t + noise
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * process_noise
    R = np.array([[measurement_noise]])

    # Initialise
    x = np.array([closes[0], 0.0])
    P = np.eye(2) * 1.0

    for i in range(n):
        y = closes[i]
        if np.isnan(y):
            level[i] = np.nan
            slope[i] = np.nan
            error[i] = np.nan
            continue

        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Innovation
        innov = y - (H @ x_pred)[0]

        # Innovation covariance
        S = (H @ P_pred @ H.T + R)[0, 0]

        # Kalman gain
        K = (P_pred @ H.T) / S  # (2, 1)

        # Update
        x = x_pred + K[:, 0] * innov
        P = P_pred - np.outer(K[:, 0], H[0]) @ P_pred

        level[i] = x[0]
        slope[i] = x[1]
        error[i] = innov

    return level, slope, error
