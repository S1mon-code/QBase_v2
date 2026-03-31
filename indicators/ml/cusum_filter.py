import numpy as np


def cusum_event_filter(
    closes: np.ndarray,
    threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CUSUM event-driven sampling.

    Identifies 'events' when cumulative deviation of returns exceeds a
    threshold (in units of rolling standard deviation).  Useful for
    filtering bars to only trade on significant price moves.

    Parameters
    ----------
    closes : (N,) price series.
    threshold : CUSUM threshold in std-dev units.

    Returns
    -------
    event_signal : (N,) +1 for positive event, -1 for negative, 0 otherwise.
    cusum_value : (N,) current CUSUM value (positive branch).
    time_since_event : (N,) bars since last event.
    """
    n = len(closes)
    event_signal = np.zeros(n, dtype=np.float64)
    cusum_value = np.full(n, np.nan, dtype=np.float64)
    time_since_event = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return event_signal, cusum_value, time_since_event

    # Log returns
    safe = np.maximum(closes, 1e-12)
    log_ret = np.full(n, np.nan, dtype=np.float64)
    log_ret[1:] = np.log(safe[1:]) - np.log(safe[:-1])

    s_pos = 0.0
    s_neg = 0.0
    last_event = 0

    for i in range(1, n):
        r = log_ret[i]
        if np.isnan(r):
            cusum_value[i] = s_pos
            time_since_event[i] = float(i - last_event)
            continue

        # Running std (use Welford-like update for efficiency)
        # Simple: use std of returns up to this bar
        if i < 20:
            std_r = np.nanstd(log_ret[1 : i + 1])
        else:
            std_r = np.nanstd(log_ret[i - 19 : i + 1])

        if std_r < 1e-12:
            cusum_value[i] = s_pos
            time_since_event[i] = float(i - last_event)
            continue

        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        cusum_value[i] = s_pos

        thresh_val = threshold * std_r

        if s_pos > thresh_val:
            event_signal[i] = 1.0
            s_pos = 0.0
            last_event = i
        elif s_neg < -thresh_val:
            event_signal[i] = -1.0
            s_neg = 0.0
            last_event = i

        time_since_event[i] = float(i - last_event)

    return event_signal, cusum_value, time_since_event
