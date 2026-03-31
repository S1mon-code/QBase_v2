import numpy as np


def runs_test(closes: np.ndarray, period: int = 60) -> tuple:
    """Wald-Wolfowitz runs test for randomness.

    Counts the number of runs (consecutive sequences of same-sign returns)
    and compares to expected under randomness.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    period : int
        Rolling window for the test.

    Returns
    -------
    z_statistic : np.ndarray
        Z-statistic of the runs test. |z|>2 = non-random.
    is_trending : np.ndarray (bool)
        True if z < -2 (fewer runs than expected = trending).
    is_mean_reverting : np.ndarray (bool)
        True if z > 2 (more runs than expected = mean reverting).
    """
    n = len(closes)
    if n == 0:
        return (np.array([], dtype=float), np.array([], dtype=bool),
                np.array([], dtype=bool))

    z_stat = np.full(n, np.nan)
    is_trend = np.zeros(n, dtype=bool)
    is_mr = np.zeros(n, dtype=bool)

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    for i in range(period, n):
        window = rets[i - period + 1:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < 20:
            continue

        # Classify as positive or negative (skip zeros)
        signs = np.sign(valid)
        nonzero = signs[signs != 0]
        if len(nonzero) < 10:
            continue

        n_pos = np.sum(nonzero > 0)
        n_neg = np.sum(nonzero < 0)
        n_total = len(nonzero)

        if n_pos == 0 or n_neg == 0:
            continue

        # Count runs
        runs = 1
        for j in range(1, len(nonzero)):
            if nonzero[j] != nonzero[j - 1]:
                runs += 1

        # Expected runs and variance under H0
        expected = 1.0 + (2.0 * n_pos * n_neg) / n_total
        var = ((2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n_total))
               / (n_total * n_total * (n_total - 1.0)))

        if var > 1e-9:
            z = (runs - expected) / np.sqrt(var)
            z_stat[i] = z
            is_trend[i] = z < -2.0
            is_mr[i] = z > 2.0

    return z_stat, is_trend, is_mr
