import numpy as np


def quantile_regression_bands(
    closes: np.ndarray,
    period: int = 60,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling quantile regression bands (non-parametric confidence intervals).

    Computes rolling empirical quantiles of the price series to form
    dynamic support/resistance bands.  Unlike Bollinger Bands these make
    no normality assumption.

    Parameters
    ----------
    closes : (N,) price series.
    period : rolling window length.
    quantiles : (lower, median, upper) quantile levels.

    Returns
    -------
    lower_band : (N,) lower quantile level.
    median : (N,) median level.
    upper_band : (N,) upper quantile level.
    """
    n = len(closes)
    lower = np.full(n, np.nan, dtype=np.float64)
    median = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return lower, median, upper

    q_lo, q_med, q_hi = quantiles

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        if np.any(np.isnan(window)):
            continue
        lower[i] = np.percentile(window, q_lo * 100)
        median[i] = np.percentile(window, q_med * 100)
        upper[i] = np.percentile(window, q_hi * 100)

    return lower, median, upper
