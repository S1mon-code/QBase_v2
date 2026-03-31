import numpy as np


def tail_dependence(returns_a, returns_b, period=120, quantile=0.05):
    """Empirical tail dependence coefficient between two return series.

    For each rolling window, computes the probability that one series is in
    its extreme tail *given* the other is also in its extreme tail.

    Parameters
    ----------
    returns_a, returns_b : 1-D return arrays of equal length.
    period : rolling window size.
    quantile : tail quantile (e.g. 0.05 = bottom/top 5%).

    Returns
    -------
    lower_tail_dep : (N,) lower tail dependence (crash together).
    upper_tail_dep : (N,) upper tail dependence (rally together).
    """
    returns_a = np.asarray(returns_a, dtype=np.float64)
    returns_b = np.asarray(returns_b, dtype=np.float64)
    n = len(returns_a)
    lower_tail_dep = np.full(n, np.nan, dtype=np.float64)
    upper_tail_dep = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return lower_tail_dep, upper_tail_dep

    for i in range(period - 1, n):
        a = returns_a[i - period + 1: i + 1]
        b = returns_b[i - period + 1: i + 1]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue

        # lower tail
        q_a_low = np.percentile(a, quantile * 100)
        q_b_low = np.percentile(b, quantile * 100)
        a_low = a <= q_a_low
        b_low = b <= q_b_low
        n_b_low = b_low.sum()
        if n_b_low > 0:
            lower_tail_dep[i] = (a_low & b_low).sum() / n_b_low
        else:
            lower_tail_dep[i] = 0.0

        # upper tail
        q_a_high = np.percentile(a, (1 - quantile) * 100)
        q_b_high = np.percentile(b, (1 - quantile) * 100)
        a_high = a >= q_a_high
        b_high = b >= q_b_high
        n_b_high = b_high.sum()
        if n_b_high > 0:
            upper_tail_dep[i] = (a_high & b_high).sum() / n_b_high
        else:
            upper_tail_dep[i] = 0.0

    return lower_tail_dep, upper_tail_dep
