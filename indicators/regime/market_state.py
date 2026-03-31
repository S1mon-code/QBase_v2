import numpy as np


def market_state(
    closes: np.ndarray,
    volumes: np.ndarray,
    oi: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-dimensional market state classifier (0-4).

    Classifies each bar into one of five states based on trend, volatility,
    and volume/OI signals:
        0 = quiet (low vol, no trend)
        1 = trending_up (positive trend + momentum)
        2 = trending_down (negative trend + momentum)
        3 = volatile_range (high vol, no clear direction)
        4 = breakout (vol expansion + directional move)

    Returns (state, state_confidence). Confidence is 0-1.
    """
    n = len(closes)
    state = np.full(n, np.nan, dtype=np.float64)
    state_confidence = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return state, state_confidence

    safe = np.maximum(closes, 1e-9)
    log_ret = np.diff(np.log(safe))

    for i in range(period, n):
        # --- Trend measure: rolling return over period ---
        ret_window = log_ret[max(0, i - period) : i]
        if len(ret_window) < 2:
            continue

        cumret = np.sum(ret_window)
        vol = np.std(ret_window, ddof=1)

        # --- Efficiency ratio ---
        net = abs(closes[i] - closes[max(0, i - period)])
        path = np.sum(np.abs(np.diff(closes[max(0, i - period) : i + 1])))
        er = net / (path + 1e-14)

        # --- Volatility expansion: current vol vs longer lookback ---
        long_start = max(0, i - period * 3)
        long_ret = log_ret[long_start : i]
        if len(long_ret) < period:
            long_vol = vol
        else:
            long_vol = np.std(long_ret, ddof=1)
        vol_ratio = vol / (long_vol + 1e-14)

        # --- Volume surge ---
        vol_window = volumes[max(0, i - period) : i]
        avg_vol = np.mean(vol_window) if len(vol_window) > 0 else 1.0
        cur_vol = volumes[i] if i < len(volumes) else avg_vol
        vol_surge = cur_vol / (avg_vol + 1e-14)

        # --- Classification logic ---
        is_trending = er > 0.4
        is_vol_high = vol_ratio > 1.3
        is_vol_low = vol_ratio < 0.7
        is_vol_surge = vol_surge > 1.5
        trend_up = cumret > 0
        trend_down = cumret < 0

        scores = np.zeros(5)

        # State 0: quiet
        if is_vol_low and not is_trending:
            scores[0] = 0.5 + 0.5 * (1.0 - vol_ratio)

        # State 1: trending up
        if trend_up and is_trending:
            scores[1] = er * min(vol_ratio, 2.0) / 2.0

        # State 2: trending down
        if trend_down and is_trending:
            scores[2] = er * min(vol_ratio, 2.0) / 2.0

        # State 3: volatile range
        if is_vol_high and not is_trending:
            scores[3] = vol_ratio / 3.0

        # State 4: breakout
        if is_vol_high and is_trending and is_vol_surge:
            scores[4] = er * vol_ratio * min(vol_surge, 3.0) / 6.0

        best_state = int(np.argmax(scores))
        total = np.sum(scores)
        conf = scores[best_state] / total if total > 1e-14 else 0.0

        state[i] = float(best_state)
        state_confidence[i] = float(np.clip(conf, 0.0, 1.0))

    return state, state_confidence
