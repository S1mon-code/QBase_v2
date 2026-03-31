import numpy as np


def inventory_proxy(closes: np.ndarray, oi: np.ndarray,
                    volumes: np.ndarray, period: int = 40) -> tuple:
    """Inventory/warehouse change proxy from basis behavior + OI.

    Without actual warehouse data, we infer inventory changes from:
    - Contango (far > near) typically implies high inventory
    - Backwardation (near > far) typically implies low inventory
    - Proxy: OI structure + price momentum as inventory signal

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volumes.
    period : int
        Lookback period for trend computation.

    Returns
    -------
    inventory_change_proxy : np.ndarray
        Positive = inventory likely building, negative = drawdown.
    inventory_zscore : np.ndarray
        Z-score of the inventory proxy over rolling window.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    inv_proxy = np.full(n, np.nan)
    inv_z = np.full(n, np.nan)

    if n <= period:
        return inv_proxy, inv_z

    # Price trend (negative trend + rising OI = inventory building)
    price_trend = np.full(n, np.nan)
    price_trend[period:] = closes[period:] / closes[:-period] - 1.0

    # OI trend
    oi_trend = np.full(n, np.nan)
    oi_trend[period:] = (oi[period:] - oi[:-period]) / np.maximum(oi[:-period], 1.0)

    # Volume trend (declining volume + stable OI = inventory accumulation)
    vol_ma_short = np.full(n, np.nan)
    vol_ma_long = np.full(n, np.nan)
    short_p = max(5, period // 4)
    for i in range(short_p, n):
        w = volumes[i - short_p:i + 1]
        valid = w[np.isfinite(w)]
        if len(valid) > 0:
            vol_ma_short[i] = np.mean(valid)
    for i in range(period, n):
        w = volumes[i - period:i + 1]
        valid = w[np.isfinite(w)]
        if len(valid) > 0:
            vol_ma_long[i] = np.mean(valid)

    for i in range(period, n):
        pt = price_trend[i]
        ot = oi_trend[i]
        vs = vol_ma_short[i]
        vl = vol_ma_long[i]

        if np.isnan(pt) or np.isnan(ot):
            continue

        # Inventory proxy components:
        # 1. Weak price + rising OI = supply building (contango-like)
        # 2. Strong price + falling OI = supply drawing (backwardation-like)
        price_component = -pt  # negative price trend = building
        oi_component = ot  # rising OI = more hedging = more inventory

        # Volume component: low turnover = stable inventory
        vol_component = 0.0
        if np.isfinite(vs) and np.isfinite(vl) and vl > 0:
            vol_component = -(vs / vl - 1.0)  # declining volume = building

        inv_proxy[i] = price_component * 0.4 + oi_component * 0.4 + vol_component * 0.2

    # Z-score
    lookback = period * 3
    for i in range(lookback, n):
        window = inv_proxy[i - lookback:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < 10:
            continue
        mu = np.mean(valid)
        std = np.std(valid)
        if std > 0 and np.isfinite(inv_proxy[i]):
            inv_z[i] = (inv_proxy[i] - mu) / std

    return inv_proxy, inv_z
