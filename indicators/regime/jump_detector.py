import numpy as np


def jump_detection(
    closes: np.ndarray,
    period: int = 20,
    threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect price jumps (returns > threshold * rolling_std).

    A jump is flagged when the absolute log return exceeds `threshold`
    times the rolling standard deviation of returns.

    Returns (is_jump, jump_size, jump_direction).
    is_jump: 1.0 if jump detected, 0.0 otherwise.
    jump_size: standardized jump magnitude (in units of rolling std).
    jump_direction: 1 = up, -1 = down, 0 = no jump.
    """
    n = len(closes)
    is_jump = np.full(n, np.nan, dtype=np.float64)
    jump_size = np.full(n, np.nan, dtype=np.float64)
    jump_direction = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return is_jump, jump_size, jump_direction

    safe = np.maximum(closes, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    if len(log_ret) < period:
        return is_jump, jump_size, jump_direction

    for i in range(period, len(log_ret)):
        # Rolling std from prior returns (exclude current to avoid look-ahead)
        window = log_ret[i - period : i]
        roll_std = np.std(window, ddof=1)

        cur_ret = log_ret[i]

        # Map log_ret index i back to closes index i+1
        idx = i + 1

        if roll_std < 1e-14:
            is_jump[idx] = 0.0
            jump_size[idx] = 0.0
            jump_direction[idx] = 0.0
            continue

        z = abs(cur_ret) / roll_std

        if z > threshold:
            is_jump[idx] = 1.0
            jump_size[idx] = z
            jump_direction[idx] = 1.0 if cur_ret > 0 else -1.0
        else:
            is_jump[idx] = 0.0
            jump_size[idx] = z
            jump_direction[idx] = 0.0

    return is_jump, jump_size, jump_direction
