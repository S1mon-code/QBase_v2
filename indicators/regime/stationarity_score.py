import numpy as np


def stationarity(closes: np.ndarray, period: int = 60) -> tuple:
    """Rolling ADF-like stationarity test score.

    Estimates a Dickey-Fuller t-statistic by regressing price changes
    on lagged levels. Stationary series = mean reverting.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    period : int
        Rolling window for the test.

    Returns
    -------
    adf_score : np.ndarray
        ADF-like t-statistic. More negative = more stationary.
        Roughly: < -2.86 at 5% significance for stationarity.
    is_stationary : np.ndarray (bool)
        True if adf_score < -2.86 (stationary at ~5% level).
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    adf = np.full(n, np.nan)
    is_stat = np.zeros(n, dtype=bool)

    for i in range(period, n):
        window = closes[i - period:i + 1]
        if not np.all(np.isfinite(window)):
            valid_mask = np.isfinite(window)
            if np.sum(valid_mask) < period // 2:
                continue
            # Skip if too many NaN
            continue

        # Dickey-Fuller regression: delta_y = alpha + beta * y_{t-1} + eps
        y = window[1:] - window[:-1]  # first differences
        x = window[:-1]  # lagged levels

        n_obs = len(y)
        if n_obs < 10:
            continue

        # OLS: y = a + b*x
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx < 1e-12:
            continue

        beta = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
        alpha = y_mean - beta * x_mean

        # Residuals and standard error
        residuals = y - alpha - beta * x
        sse = np.sum(residuals ** 2)
        se_beta = np.sqrt(sse / (n_obs - 2) / ss_xx)

        if se_beta > 1e-12:
            t_stat = beta / se_beta
            adf[i] = t_stat
            is_stat[i] = t_stat < -2.86

    return adf, is_stat
