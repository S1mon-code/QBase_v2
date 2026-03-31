import numpy as np


def oi_mean_reversion(oi: np.ndarray, period: int = 60) -> tuple:
    """OI mean-reversion signal.

    Extreme OI levels tend to revert to the mean.  This indicator
    provides a z-score, estimates the half-life of OI mean reversion
    via OLS on lagged OI, and outputs a directional reversion signal.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    period : int
        Lookback for z-score and half-life estimation.

    Returns
    -------
    oi_zscore : np.ndarray
        Rolling z-score of OI.
    half_life : np.ndarray
        Estimated mean-reversion half-life (bars). NaN if not
        mean-reverting.
    reversion_signal : np.ndarray
        -1 when OI is extremely high (expect drop), +1 when
        extremely low (expect rise), 0 otherwise.
    """
    n = len(oi)
    oi_zscore = np.full(n, np.nan)
    half_life = np.full(n, np.nan)
    reversion_signal = np.zeros(n, dtype=float)

    if n < period:
        return oi_zscore, half_life, reversion_signal

    for i in range(period - 1, n):
        window = oi[i - period + 1:i + 1]
        m = np.mean(window)
        s = np.std(window, ddof=1)

        if s > 0:
            z = (oi[i] - m) / s
            oi_zscore[i] = z
        else:
            oi_zscore[i] = 0.0
            z = 0.0

        # Half-life via OLS: delta_oi = beta * oi_lag + alpha
        delta = np.diff(window)
        lag = window[:-1]
        lag_dm = lag - np.mean(lag)
        denom = np.dot(lag_dm, lag_dm)
        if denom > 0:
            beta = np.dot(lag_dm, delta) / denom
            if beta < 0:
                hl = -np.log(2) / beta
                half_life[i] = hl
            # else: not mean-reverting, leave NaN

        # Signal
        if z > 2.0:
            reversion_signal[i] = -1.0
        elif z < -2.0:
            reversion_signal[i] = 1.0

    return oi_zscore, half_life, reversion_signal
