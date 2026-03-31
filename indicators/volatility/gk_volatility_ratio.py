import numpy as np


def gk_vol_ratio(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    fast: int = 10,
    slow: int = 60,
) -> tuple:
    """Garman-Klass volatility ratio: fast GK vol / slow GK vol.

    Uses the Garman-Klass estimator for more efficient volatility estimation
    from OHLC data. Ratio > 1 = volatility expanding, < 1 = contracting.
    Returns (ratio, regime) where regime: 1=expanding, 0=neutral, -1=contracting.
    """
    n = len(closes)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan)

    opens = opens.astype(np.float64)
    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)

    # Per-bar GK variance component
    # GK = 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
    log_hl = np.log(highs / lows)
    log_co = np.log(closes / opens)
    gk_var = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2

    # Rolling GK volatility
    fast_vol = np.full(n, np.nan)
    slow_vol = np.full(n, np.nan)

    for i in range(fast - 1, n):
        fast_vol[i] = np.sqrt(np.mean(gk_var[i - fast + 1 : i + 1]))

    for i in range(slow - 1, n):
        slow_vol[i] = np.sqrt(np.mean(gk_var[i - slow + 1 : i + 1]))

    ratio = np.full(n, np.nan)
    valid = (~np.isnan(fast_vol)) & (~np.isnan(slow_vol)) & (slow_vol > 0)
    ratio[valid] = fast_vol[valid] / slow_vol[valid]

    # Regime classification
    regime = np.full(n, np.nan)
    regime[valid & (ratio > 1.2)] = 1.0   # expanding
    regime[valid & (ratio < 0.8)] = -1.0  # contracting
    regime[valid & (ratio >= 0.8) & (ratio <= 1.2)] = 0.0  # neutral

    return ratio, regime
