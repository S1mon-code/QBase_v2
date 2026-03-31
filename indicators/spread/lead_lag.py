"""Lead-lag detector via rolling cross-correlation.

Determines which of two assets tends to move first (the leader) and
by how many bars.
"""

import numpy as np


def lead_lag(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    max_lag: int = 10,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect which asset leads/lags using rolling cross-correlation.

    Parameters
    ----------
    closes_a : close prices of asset A.
    closes_b : close prices of asset B.
    max_lag  : maximum lag (in bars) to test in each direction.
    period   : rolling window for cross-correlation estimation.

    Returns
    -------
    (lead_lag_score, optimal_lag)
        lead_lag_score – positive means A leads B; negative means B
                         leads A.  Magnitude reflects strength.
        optimal_lag    – the lag (in bars) with highest absolute
                         cross-correlation.  Positive = A leads.
    """
    n = len(closes_a)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    safe_a = np.where(closes_a == 0, np.nan, closes_a)
    safe_b = np.where(closes_b == 0, np.nan, closes_b)
    ret_a = np.full(n, np.nan)
    ret_b = np.full(n, np.nan)
    ret_a[1:] = safe_a[1:] / safe_a[:-1] - 1.0
    ret_b[1:] = safe_b[1:] / safe_b[:-1] - 1.0

    score = np.full(n, np.nan)
    opt_lag = np.full(n, np.nan)

    min_start = period + max_lag
    for i in range(min_start, n):
        best_corr = 0.0
        best_lag = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            # lag > 0: A at time t vs B at time t+lag  (A leads)
            # lag < 0: B at time t vs A at time t+|lag| (B leads)
            if lag > 0:
                ra = ret_a[i - period + 1 - lag : i + 1 - lag]
                rb = ret_b[i - period + 1 : i + 1]
            else:
                ra = ret_a[i - period + 1 : i + 1]
                rb = ret_b[i - period + 1 + lag : i + 1 + lag]

            mask = ~(np.isnan(ra) | np.isnan(rb))
            if np.sum(mask) < 10:
                continue
            va, vb = ra[mask], rb[mask]
            std_a = np.std(va, ddof=1)
            std_b = np.std(vb, ddof=1)
            if std_a == 0 or std_b == 0:
                continue
            c = np.corrcoef(va, vb)[0, 1]
            if abs(c) > abs(best_corr):
                best_corr = c
                best_lag = lag

        score[i] = best_corr * np.sign(best_lag) if best_lag != 0 else 0.0
        opt_lag[i] = float(best_lag)

    return score, opt_lag
