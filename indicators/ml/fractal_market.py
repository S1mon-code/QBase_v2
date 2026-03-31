import numpy as np


def fractal_market_hypothesis(
    closes: np.ndarray,
    period: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fractal Market Hypothesis features.

    Computes multi-scale Hurst exponents to assess market stability.
    When all horizons show similar Hurst, the market is stable; divergent
    Hurst across scales signals instability.

    Parameters
    ----------
    closes : (N,) price series.
    period : analysis window length.

    Returns
    -------
    stability_index : (N,) 0-1 measure (1 = all scales agree, 0 = divergent).
    dominant_horizon : (N,) scale index with strongest trend (highest Hurst).
    is_stable : (N,) 1.0 if stability_index > 0.7, else 0.0.
    """
    n = len(closes)
    stability = np.full(n, np.nan, dtype=np.float64)
    dominant = np.full(n, np.nan, dtype=np.float64)
    is_stable = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return stability, dominant, is_stable

    scales = [5, 10, 20, 40]
    scales = [s for s in scales if s < period // 3]
    if len(scales) < 2:
        return stability, dominant, is_stable

    safe = np.maximum(closes, 1e-12)

    for i in range(period, n):
        window = safe[i - period : i]
        if np.any(np.isnan(window)):
            continue

        log_p = np.log(window)
        hursts = []

        for scale in scales:
            h = _rescaled_range_hurst(log_p, scale)
            if not np.isnan(h):
                hursts.append(h)

        if len(hursts) < 2:
            continue

        hursts_arr = np.array(hursts)

        # Stability = 1 - normalised std of Hurst exponents
        h_std = np.std(hursts_arr)
        stability[i] = max(0.0, 1.0 - 2.0 * h_std)

        # Dominant horizon = scale with highest Hurst
        dominant[i] = float(scales[np.argmax(hursts_arr)])

        is_stable[i] = 1.0 if stability[i] > 0.7 else 0.0

    return stability, dominant, is_stable


def _rescaled_range_hurst(log_prices: np.ndarray, scale: int) -> float:
    """Estimate Hurst exponent via rescaled range at a given scale."""
    n = len(log_prices)
    returns = log_prices[1:] - log_prices[:-1]
    n_ret = len(returns)

    if n_ret < scale * 2:
        return np.nan

    n_segments = n_ret // scale
    if n_segments < 2:
        return np.nan

    rs_values = []
    for seg in range(n_segments):
        chunk = returns[seg * scale : (seg + 1) * scale]
        mean_r = np.mean(chunk)
        deviations = np.cumsum(chunk - mean_r)
        r = np.max(deviations) - np.min(deviations)
        s = np.std(chunk)
        if s > 1e-12:
            rs_values.append(r / s)

    if len(rs_values) < 1:
        return np.nan

    mean_rs = np.mean(rs_values)
    if mean_rs <= 0:
        return np.nan

    # Hurst = log(R/S) / log(n)
    h = np.log(mean_rs) / np.log(scale)
    return np.clip(h, 0.0, 1.0)
