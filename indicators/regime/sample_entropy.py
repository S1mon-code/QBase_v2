import numpy as np


def sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r_mult: float = 0.2,
    period: int = 60,
) -> np.ndarray:
    """Sample entropy (complexity/regularity measure).

    Measures the conditional probability that sequences similar for m points
    remain similar at m+1 points, within tolerance r = r_mult * std(window).

    Returns entropy_value. Low = regular/predictable, High = random.
    """
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return out

    safe = np.maximum(data, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    if len(log_ret) < period:
        return out

    for i in range(period, len(log_ret) + 1):
        window = log_ret[i - period : i]
        w_len = len(window)

        std = np.std(window, ddof=1)
        if std < 1e-14:
            out[i] = 0.0
            continue

        r = r_mult * std
        out[i] = _sampen(window, m, r)

    return out


def _sampen(x: np.ndarray, m: int, r: float) -> float:
    """Compute sample entropy for a single window."""
    n = len(x)
    if n < m + 2:
        return 0.0

    # Count template matches for dimension m and m+1
    count_m = 0
    count_m1 = 0

    for i in range(n - m):
        for j in range(i + 1, n - m):
            # Check m-length match (Chebyshev distance)
            if np.max(np.abs(x[i : i + m] - x[j : j + m])) <= r:
                count_m += 1
                # Check (m+1)-length match
                if i + m < n and j + m < n:
                    if abs(x[i + m] - x[j + m]) <= r:
                        count_m1 += 1

    if count_m == 0 or count_m1 == 0:
        return 0.0

    return -np.log(count_m1 / count_m)
