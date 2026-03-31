import numpy as np


def adaptive_kalman(
    closes: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kalman filter with adaptive noise parameters estimated from data.

    Uses a rolling window to estimate the process noise and measurement
    noise from recent prediction errors, making the filter self-tuning.

    Parameters
    ----------
    closes : (N,) price series.
    period : window length for noise estimation.

    Returns
    -------
    trend : (N,) filtered price level.
    momentum : (N,) filtered slope (trend direction / speed).
    volatility_estimate : (N,) estimated measurement noise (market vol).
    """
    n = len(closes)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    trend = np.full(n, np.nan, dtype=np.float64)
    momentum = np.full(n, np.nan, dtype=np.float64)
    vol_est = np.full(n, np.nan, dtype=np.float64)

    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    # Initial guesses
    q = 0.01
    r = 1.0
    Q = np.eye(2) * q
    R = np.array([[r]])

    x = np.array([closes[0] if not np.isnan(closes[0]) else 0.0, 0.0])
    P = np.eye(2) * 1.0

    # Store innovations for adaptive estimation
    innovations = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        y = closes[i]
        if np.isnan(y):
            trend[i] = np.nan
            momentum[i] = np.nan
            vol_est[i] = np.nan
            continue

        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Innovation
        innov = y - (H @ x_pred)[0]
        innovations[i] = innov

        S = (H @ P_pred @ H.T + R)[0, 0]
        K = (P_pred @ H.T) / S

        # Update state
        x = x_pred + K[:, 0] * innov
        P = P_pred - np.outer(K[:, 0], H[0]) @ P_pred

        trend[i] = x[0]
        momentum[i] = x[1]

        # Adaptive noise estimation from recent innovations
        if i >= period:
            recent = innovations[i - period + 1 : i + 1]
            valid = recent[~np.isnan(recent)]
            if len(valid) > 5:
                innov_var = np.var(valid)
                # R ≈ innovation variance - H @ P_pred @ H^T
                r_new = max(1e-6, innov_var - (H @ P_pred @ H.T)[0, 0])
                R = np.array([[r_new]])
                # Q scales with innovation variance
                q_new = max(1e-6, innov_var * 0.01)
                Q = np.eye(2) * q_new
                vol_est[i] = np.sqrt(r_new)
            else:
                vol_est[i] = np.nan
        else:
            vol_est[i] = np.nan

    return trend, momentum, vol_est
