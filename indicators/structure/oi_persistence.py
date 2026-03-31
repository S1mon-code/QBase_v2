import numpy as np


def oi_persistence(oi: np.ndarray, period: int = 20) -> tuple:
    """Persistence (autocorrelation) of OI changes.

    High persistence means OI is consistently building or unwinding
    (sustained position trend).  Low persistence means OI changes
    are noisy and mean-reverting.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    period : int
        Lookback for autocorrelation.

    Returns
    -------
    persistence : np.ndarray
        Lag-1 autocorrelation of OI changes over *period*.
        Range roughly [-1, 1].
    trend_strength : np.ndarray
        Proportion of OI changes in the dominant direction.
        Range [0.5, 1.0].
    """
    n = len(oi)
    persistence = np.full(n, np.nan)
    trend_strength = np.full(n, np.nan)

    if n < period + 2:
        return persistence, trend_strength

    oi_change = np.diff(oi)  # length n-1

    for i in range(period + 1, n):
        # oi_change indices: i-1 corresponds to oi[i] - oi[i-1]
        chg_win = oi_change[i - period:i]  # last *period* changes

        # Lag-1 autocorrelation
        x = chg_win[:-1]
        y = chg_win[1:]
        mx = np.mean(x)
        my = np.mean(y)
        dx = x - mx
        dy = y - my
        denom = np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))
        if denom > 0:
            persistence[i] = np.sum(dx * dy) / denom
        else:
            persistence[i] = 0.0

        # Trend strength: fraction of changes in dominant direction
        n_pos = np.sum(chg_win > 0)
        n_neg = np.sum(chg_win < 0)
        total = n_pos + n_neg
        if total > 0:
            trend_strength[i] = max(n_pos, n_neg) / total
        else:
            trend_strength[i] = 0.5

    return persistence, trend_strength
