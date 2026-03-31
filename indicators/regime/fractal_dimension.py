import numpy as np


def fractal_dim(
    data: np.ndarray,
    period: int = 60,
) -> np.ndarray:
    """Higuchi fractal dimension estimator.

    Computes the fractal dimension of the price series within rolling windows.
    Uses k_max = period // 4 interval sizes.

    Returns dimension. ~1.0 = smooth trend, ~1.5 = random walk, ~2.0 = space-filling.
    """
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return out

    safe = np.maximum(data, 1e-9)
    log_p = np.log(safe)

    k_max = max(2, period // 4)

    for i in range(period, n + 1):
        window = log_p[i - period : i]
        out[i - 1] = _higuchi(window, k_max)

    return out


def _higuchi(x: np.ndarray, k_max: int) -> float:
    """Higuchi fractal dimension for a single window."""
    n = len(x)
    if n < 4:
        return 1.5

    log_k = []
    log_lk = []

    for k in range(1, k_max + 1):
        lengths = []
        for m in range(1, k + 1):
            # Subsequence: x[m-1], x[m-1+k], x[m-1+2k], ...
            idx = np.arange(m - 1, n, k)
            if len(idx) < 2:
                continue
            subseq = x[idx]
            # Normalized length
            seg_len = np.sum(np.abs(np.diff(subseq)))
            norm = (n - 1) / (k * len(idx) * k) if len(idx) > 0 else 0
            # Higuchi normalization: L_m(k) = sum|diff| * (N-1) / (floor((N-m)/k) * k)
            num_intervals = len(subseq) - 1
            if num_intervals > 0:
                lk = seg_len * (n - 1) / (num_intervals * k * k)
                lengths.append(lk)

        if len(lengths) > 0:
            avg_lk = np.mean(lengths)
            if avg_lk > 1e-14:
                log_k.append(np.log(1.0 / k))
                log_lk.append(np.log(avg_lk))

    if len(log_k) < 2:
        return 1.5

    # Linear fit: log(L(k)) vs log(1/k), slope = fractal dimension
    slope = np.polyfit(np.array(log_k), np.array(log_lk), 1)[0]
    return float(np.clip(slope, 1.0, 2.0))
