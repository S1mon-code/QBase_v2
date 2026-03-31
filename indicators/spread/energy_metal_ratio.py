"""Energy/Metal ratio for macro regime detection (e.g. oil/copper).

A rising ratio suggests energy outperforming metals, often signalling
inflation or supply-driven regime.  Z-score flags extremes.
"""

import numpy as np


def energy_metal_ratio(
    energy_closes: np.ndarray,
    metal_closes: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energy/Metal price ratio with z-score and trend.

    Parameters
    ----------
    energy_closes : array of energy asset closing prices (e.g. oil).
    metal_closes  : array of metal asset closing prices (e.g. copper).
    period        : lookback window for z-score and trend calculation.

    Returns
    -------
    (ratio, zscore, trend)
        ratio   – energy / metal price ratio.
        zscore  – rolling z-score of the ratio.
        trend   – rolling slope of the ratio (positive = energy gaining).
    """
    n = len(energy_closes)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()

    ratio = np.full(n, np.nan)
    zscore = np.full(n, np.nan)
    trend = np.full(n, np.nan)

    safe_metal = np.where(metal_closes == 0, np.nan, metal_closes)
    ratio = energy_closes / safe_metal

    for i in range(period - 1, n):
        window = ratio[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            zscore[i] = (ratio[i] - mu) / sigma

        # trend = linear regression slope over the window
        x = np.arange(len(valid), dtype=float)
        x_mean = np.mean(x)
        denom = np.sum((x - x_mean) ** 2)
        if denom > 0:
            trend[i] = np.sum((x - x_mean) * (valid - np.mean(valid))) / denom

    return ratio, zscore, trend
