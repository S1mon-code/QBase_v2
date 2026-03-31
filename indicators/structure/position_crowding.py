import numpy as np


def position_crowding(closes: np.ndarray, oi: np.ndarray,
                      volumes: np.ndarray, period: int = 60) -> tuple:
    """Position crowding detector.

    Crowded positions occur when OI is high, volume is low (positions are
    sticky), and price is trending. These create unwind risk -- when the
    crowded trade reverses, exits are jammed.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volumes.
    period : int
        Lookback period for percentile calculations.

    Returns
    -------
    crowding_score : np.ndarray
        Position crowding score (0-1). Higher = more crowded.
    unwind_risk : np.ndarray
        Unwind risk score (0-1). High = potential squeeze on reversal.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    crowding = np.full(n, np.nan)
    unwind = np.full(n, np.nan)

    for i in range(period, n):
        window_oi = oi[i - period:i + 1]
        window_vol = volumes[i - period:i + 1]
        window_close = closes[i - period:i + 1]

        valid_oi = window_oi[np.isfinite(window_oi)]
        valid_vol = window_vol[np.isfinite(window_vol)]

        if len(valid_oi) < 10 or len(valid_vol) < 10:
            continue

        cur_oi = oi[i]
        cur_vol = volumes[i]
        if np.isnan(cur_oi) or np.isnan(cur_vol) or cur_vol == 0:
            continue

        # OI percentile (high = lots of positions)
        oi_pctl = np.sum(valid_oi <= cur_oi) / len(valid_oi)

        # Volume percentile (low = sticky positions, inverted)
        vol_pctl = 1.0 - np.sum(valid_vol <= cur_vol) / len(valid_vol)

        # Trend strength: absolute return over period
        if closes[i - period] > 0:
            trend = abs(closes[i] / closes[i - period] - 1.0)
        else:
            trend = 0.0

        # Normalize trend to 0-1
        trend_score = min(1.0, trend / 0.20)  # 20% move = max

        # Crowding = high OI + low volume + trending
        crowding[i] = np.clip(
            (oi_pctl * 0.4 + vol_pctl * 0.3 + trend_score * 0.3),
            0.0, 1.0
        )

        # Unwind risk: crowding * inverse liquidity
        vol_oi_ratio = cur_vol / max(cur_oi, 1.0)
        vol_oi_window = valid_vol / np.maximum(valid_oi, 1.0)
        if np.std(vol_oi_window) > 0:
            liquidity_z = (vol_oi_ratio - np.mean(vol_oi_window)) / np.std(vol_oi_window)
        else:
            liquidity_z = 0.0

        # Low liquidity + high crowding = high unwind risk
        illiquidity = max(0.0, -liquidity_z) / 3.0
        unwind[i] = np.clip(crowding[i] * (0.5 + 0.5 * illiquidity), 0.0, 1.0)

    return crowding, unwind
