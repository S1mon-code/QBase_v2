import numpy as np


def net_positioning_proxy(closes: np.ndarray, oi: np.ndarray,
                          volumes: np.ndarray, period: int = 20) -> tuple:
    """Estimate net long/short positioning from price-OI-volume dynamics.

    Uses the relationship between price changes, OI changes, and volume
    to infer whether the market is net long or net short. Rising prices
    with rising OI = net long buildup; falling prices with rising OI =
    net short buildup.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volumes.
    period : int
        Smoothing period.

    Returns
    -------
    net_position_estimate : np.ndarray
        Estimated net positioning. Positive = net long, negative = net short.
        Normalized to approximately -1 to +1.
    positioning_momentum : np.ndarray
        Rate of change of net positioning. Positive = longs increasing.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    net_pos = np.full(n, np.nan)
    pos_mom = np.full(n, np.nan)

    if n < 2:
        return net_pos, pos_mom

    # Daily changes
    price_ret = np.empty(n)
    price_ret[0] = 0.0
    price_ret[1:] = closes[1:] / closes[:-1] - 1.0

    oi_chg = np.empty(n)
    oi_chg[0] = 0.0
    oi_chg[1:] = oi[1:] - oi[:-1]

    # Normalize OI change by current OI level
    oi_chg_norm = np.zeros(n)
    for i in range(1, n):
        if oi[i - 1] > 0 and np.isfinite(oi_chg[i]):
            oi_chg_norm[i] = oi_chg[i] / oi[i - 1]

    # Net positioning signal: price_return * sign(oi_change) * volume_weight
    # Positive price + positive OI change = net long buildup
    # Negative price + positive OI change = net short buildup
    raw_signal = np.zeros(n)
    for i in range(1, n):
        if np.isfinite(price_ret[i]) and np.isfinite(oi_chg_norm[i]):
            # Weight by OI change magnitude
            weight = abs(oi_chg_norm[i])
            raw_signal[i] = price_ret[i] * np.sign(oi_chg_norm[i]) * (1.0 + weight * 10.0)

    # Cumulative and smoothed net positioning
    cum_signal = np.cumsum(raw_signal)

    for i in range(period - 1, n):
        window = cum_signal[i - period + 1:i + 1]
        # De-trend: subtract linear trend to get relative positioning
        x = np.arange(period, dtype=float)
        slope = (window[-1] - window[0]) / max(period - 1, 1)
        detrended = window - (window[0] + slope * x)
        net_pos[i] = detrended[-1]

    # Normalize to -1, +1
    valid_pos = net_pos[np.isfinite(net_pos)]
    if len(valid_pos) > 10:
        std = np.std(valid_pos)
        if std > 0:
            net_pos = np.where(np.isfinite(net_pos),
                               np.clip(net_pos / (3.0 * std), -1.0, 1.0),
                               np.nan)

    # Positioning momentum: change in net positioning
    for i in range(period, n):
        if np.isfinite(net_pos[i]) and np.isfinite(net_pos[i - period // 2]):
            pos_mom[i] = net_pos[i] - net_pos[i - period // 2]

    return net_pos, pos_mom
