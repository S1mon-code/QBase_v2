import numpy as np


def oi_price_regime(closes: np.ndarray, oi: np.ndarray,
                    period: int = 20) -> tuple:
    """4-state regime from OI change + price change.

    Classifies market structure into:
        1 = new_longs     (price up, OI up)
        2 = short_covering (price up, OI down)
        3 = new_shorts    (price down, OI up)
        4 = long_liquidation (price down, OI down)

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    period : int
        Lookback period for change computation.

    Returns
    -------
    regime : np.ndarray (int)
        Regime label 1-4. 0 during warmup.
    regime_strength : np.ndarray
        Magnitude of the regime signal. Higher = more decisive.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    regime = np.zeros(n, dtype=int)
    regime_strength = np.full(n, np.nan)

    if n <= period:
        return regime, regime_strength

    price_chg = np.full(n, np.nan)
    oi_chg = np.full(n, np.nan)
    price_chg[period:] = closes[period:] / closes[:-period] - 1.0
    oi_chg[period:] = oi[period:] - oi[:-period]

    for i in range(period, n):
        pc = price_chg[i]
        oc = oi_chg[i]
        if np.isnan(pc) or np.isnan(oc):
            continue

        if pc > 0 and oc > 0:
            regime[i] = 1  # new longs
        elif pc > 0 and oc <= 0:
            regime[i] = 2  # short covering
        elif pc <= 0 and oc > 0:
            regime[i] = 3  # new shorts
        else:
            regime[i] = 4  # long liquidation

        # Strength: product of normalized changes
        regime_strength[i] = np.sqrt(pc ** 2 + (oc / max(np.abs(oi[i]), 1.0)) ** 2)

    return regime, regime_strength
