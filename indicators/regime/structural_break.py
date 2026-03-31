import numpy as np


def cusum_break(
    data: np.ndarray,
    period: int = 60,
    threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CUSUM structural break detector.

    Tracks cumulative deviations from the rolling mean of log returns.
    When positive or negative CUSUM exceeds `threshold` standard deviations,
    a structural break (regime change) is flagged.

    Returns (cusum_pos, cusum_neg, break_signal).
    break_signal = 1 when CUSUM exceeds threshold (regime change detected).
    """
    n = len(data)
    cusum_pos = np.full(n, np.nan, dtype=np.float64)
    cusum_neg = np.full(n, np.nan, dtype=np.float64)
    break_signal = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return cusum_pos, cusum_neg, break_signal

    safe = np.maximum(data, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    if len(log_ret) < period:
        return cusum_pos, cusum_neg, break_signal

    for i in range(period, len(log_ret) + 1):
        window = log_ret[i - period : i]
        mu = np.mean(window)
        sigma = np.std(window, ddof=1)
        if sigma < 1e-12:
            cusum_pos[i] = 0.0
            cusum_neg[i] = 0.0
            break_signal[i] = 0.0
            continue

        # Compute CUSUM on the window
        centered = window - mu
        cs = np.cumsum(centered) / sigma

        cp = np.max(cs)
        cn = -np.min(cs)

        cusum_pos[i] = cp
        cusum_neg[i] = cn
        break_signal[i] = 1.0 if (cp > threshold or cn > threshold) else 0.0

    return cusum_pos, cusum_neg, break_signal
