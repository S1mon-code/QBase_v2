import numpy as np


def correlation_breakdown(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    period: int = 60,
    stress_threshold: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect correlation breakdown during stress.

    Compares correlation during normal periods vs stress periods
    (when either asset has returns > stress_threshold * rolling_std).
    Correlation spikes during crises indicate breakdown of diversification.

    closes_a, closes_b: price series (not returns).

    Returns (normal_corr, stress_corr, is_breakdown).
    is_breakdown = 1 when stress_corr significantly exceeds normal_corr.
    """
    n = len(closes_a)
    normal_corr = np.full(n, np.nan, dtype=np.float64)
    stress_corr = np.full(n, np.nan, dtype=np.float64)
    is_breakdown = np.full(n, np.nan, dtype=np.float64)

    if n < 2 or len(closes_b) != n:
        return normal_corr, stress_corr, is_breakdown

    safe_a = np.maximum(closes_a, 1e-9)
    safe_b = np.maximum(closes_b, 1e-9)
    ret_a = np.diff(np.log(safe_a))
    ret_b = np.diff(np.log(safe_b))

    if len(ret_a) < period:
        return normal_corr, stress_corr, is_breakdown

    for i in range(period, len(ret_a) + 1):
        win_a = ret_a[i - period : i]
        win_b = ret_b[i - period : i]

        std_a = np.std(win_a, ddof=1)
        std_b = np.std(win_b, ddof=1)

        if std_a < 1e-14 or std_b < 1e-14:
            normal_corr[i] = 0.0
            stress_corr[i] = 0.0
            is_breakdown[i] = 0.0
            continue

        # Identify stress bars
        stress_mask = (np.abs(win_a) > stress_threshold * std_a) | \
                      (np.abs(win_b) > stress_threshold * std_b)
        normal_mask = ~stress_mask

        # Full correlation
        full_corr = np.corrcoef(win_a, win_b)[0, 1]

        # Normal period correlation
        if np.sum(normal_mask) > 3:
            nc = np.corrcoef(win_a[normal_mask], win_b[normal_mask])[0, 1]
            if np.isnan(nc):
                nc = full_corr
        else:
            nc = full_corr

        # Stress period correlation
        if np.sum(stress_mask) > 3:
            sc = np.corrcoef(win_a[stress_mask], win_b[stress_mask])[0, 1]
            if np.isnan(sc):
                sc = full_corr
        else:
            sc = full_corr

        normal_corr[i] = nc
        stress_corr[i] = sc

        # Breakdown: stress correlation significantly higher than normal
        diff = abs(sc) - abs(nc)
        is_breakdown[i] = 1.0 if diff > 0.3 else 0.0

    return normal_corr, stress_corr, is_breakdown
