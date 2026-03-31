import numpy as np


def volume_efficiency(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> tuple:
    """Volume Efficiency: price movement per unit of volume.

    Measures how efficiently volume translates into price movement.
    High efficiency = trending market (price moves a lot on volume).
    Low efficiency = choppy market (volume consumed without directional movement).
    Returns (efficiency, efficiency_zscore).
    """
    n = len(closes)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan)

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    efficiency = np.full(n, np.nan)

    for i in range(period, n):
        # Net price change over period
        net_change = abs(closes[i] - closes[i - period])

        # Total volume over period
        total_vol = np.sum(volumes[i - period + 1 : i + 1])

        if total_vol > 0 and closes[i - period] > 0:
            # Normalize by starting price to make comparable across instruments
            pct_change = net_change / closes[i - period]
            efficiency[i] = pct_change / (total_vol / 1e6)  # scale volume to millions

    # Z-score
    z_lookback = period * 3
    efficiency_zscore = np.full(n, np.nan)
    for i in range(z_lookback, n):
        window = efficiency[i - z_lookback + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= period:
            mu = np.mean(valid)
            sigma = np.std(valid, ddof=1)
            if sigma > 0 and not np.isnan(efficiency[i]):
                efficiency_zscore[i] = (efficiency[i] - mu) / sigma

    return efficiency, efficiency_zscore
