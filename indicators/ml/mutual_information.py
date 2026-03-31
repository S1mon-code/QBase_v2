import numpy as np


def rolling_mutual_info(
    series_a: np.ndarray,
    series_b: np.ndarray,
    period: int = 60,
    n_bins: int = 10,
) -> np.ndarray:
    """Rolling mutual information between two series.

    Estimates MI via histogram binning on a trailing window.  MI captures
    both linear and non-linear dependencies between two variables.

    Parameters
    ----------
    series_a : (N,) first time series.
    series_b : (N,) second time series.
    period : rolling window length.
    n_bins : number of histogram bins per dimension.

    Returns
    -------
    mi_score : (N,) mutual information in nats.
        High MI = series are informatively related.
    """
    n = len(series_a)
    mi = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return mi

    for i in range(period - 1, n):
        a = series_a[i - period + 1 : i + 1]
        b = series_b[i - period + 1 : i + 1]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue

        # 2D histogram → joint distribution
        hist_2d, _, _ = np.histogram2d(a, b, bins=n_bins)
        # Normalise to probabilities
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)  # marginal of a
        py = pxy.sum(axis=0)  # marginal of b

        # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        mi_val = 0.0
        for xi in range(n_bins):
            for yi in range(n_bins):
                if pxy[xi, yi] > 1e-12 and px[xi] > 1e-12 and py[yi] > 1e-12:
                    mi_val += pxy[xi, yi] * np.log(pxy[xi, yi] / (px[xi] * py[yi]))

        mi[i] = max(0.0, mi_val)  # MI >= 0 by definition

    return mi
