import numpy as np


def speculation_index(volumes: np.ndarray, oi: np.ndarray,
                      period: int = 20) -> tuple:
    """Speculation vs hedging ratio using volume/OI as speculation proxy.

    High volume relative to OI indicates speculative activity (frequent
    position turnover). Low ratio indicates hedging (stable positions).

    Parameters
    ----------
    volumes : np.ndarray
        Trading volumes.
    oi : np.ndarray
        Open interest.
    period : int
        Smoothing / trend period.

    Returns
    -------
    spec_index : np.ndarray
        Speculation index (smoothed volume/OI ratio).
    spec_trend : np.ndarray
        Trend of speculation index (positive = speculation rising).
    is_speculative_regime : np.ndarray (bool)
        True if speculation index is above its 75th percentile over
        a longer lookback.
    """
    n = len(volumes)
    if n == 0:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=bool))

    spec_idx = np.full(n, np.nan)
    spec_trend = np.full(n, np.nan)
    is_spec = np.zeros(n, dtype=bool)

    # Raw ratio
    raw_ratio = np.full(n, np.nan)
    for i in range(n):
        if oi[i] > 0 and np.isfinite(volumes[i]) and np.isfinite(oi[i]):
            raw_ratio[i] = volumes[i] / oi[i]

    # Smoothed speculation index (SMA)
    for i in range(period - 1, n):
        window = raw_ratio[i - period + 1:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) >= 3:
            spec_idx[i] = np.mean(valid)

    # Trend: change in spec_index over period
    half_p = max(1, period // 2)
    for i in range(period - 1 + half_p, n):
        if np.isfinite(spec_idx[i]) and np.isfinite(spec_idx[i - half_p]):
            spec_trend[i] = spec_idx[i] - spec_idx[i - half_p]

    # Regime detection: is current spec_index above 75th percentile?
    long_lookback = period * 5
    for i in range(long_lookback, n):
        window = spec_idx[i - long_lookback:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < 20:
            continue
        if np.isfinite(spec_idx[i]):
            pctl_75 = np.percentile(valid, 75)
            is_spec[i] = spec_idx[i] > pctl_75

    return spec_idx, spec_trend, is_spec
