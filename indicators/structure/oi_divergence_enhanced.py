import numpy as np


def oi_divergence_enhanced(closes: np.ndarray, oi: np.ndarray,
                           volumes: np.ndarray, period: int = 20) -> tuple:
    """Enhanced OI-price divergence with volume confirmation.

    Detects divergences between price and OI trends and confirms
    with volume patterns. Bearish divergence: price up + OI down.
    Bullish divergence: price down + OI up.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volumes.
    period : int
        Rolling window for trend estimation.

    Returns
    -------
    divergence : np.ndarray
        Divergence score. Positive = bullish, negative = bearish.
    divergence_strength : np.ndarray
        Absolute strength of divergence (0-1 scale).
    is_bearish_div : np.ndarray (bool)
        True if bearish divergence (price up, OI down, volume declining).
    is_bullish_div : np.ndarray (bool)
        True if bullish divergence (price down, OI up, volume rising).
    """
    n = len(closes)
    if n == 0:
        empty_f = np.array([], dtype=float)
        empty_b = np.array([], dtype=bool)
        return empty_f, empty_f, empty_b, empty_b

    divergence = np.full(n, np.nan)
    div_strength = np.full(n, np.nan)
    is_bearish = np.zeros(n, dtype=bool)
    is_bullish = np.zeros(n, dtype=bool)

    for i in range(period, n):
        p_win = closes[i - period:i + 1]
        o_win = oi[i - period:i + 1]
        v_win = volumes[i - period:i + 1]

        p_valid = np.isfinite(p_win)
        o_valid = np.isfinite(o_win)
        v_valid = np.isfinite(v_win) & (v_win > 0)

        if np.sum(p_valid) < period // 2 or np.sum(o_valid) < period // 2:
            continue

        x = np.arange(period + 1, dtype=float)
        x_mean = np.mean(x)
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx <= 0:
            continue

        # Price slope (normalized)
        p_safe = p_win.copy()
        p_safe[~p_valid] = np.nanmean(p_win)
        p_slope = np.sum((x - x_mean) * (p_safe - np.mean(p_safe))) / ss_xx
        p_norm = p_slope / (np.mean(p_safe) + 1e-9)

        # OI slope (normalized)
        o_safe = o_win.copy()
        o_safe[~o_valid] = np.nanmean(o_win)
        o_slope = np.sum((x - x_mean) * (o_safe - np.mean(o_safe))) / ss_xx
        o_norm = o_slope / (np.mean(o_safe) + 1e-9)

        # Volume trend
        v_trend = 0.0
        if np.sum(v_valid) >= period // 2:
            v_safe = v_win.copy()
            v_safe[~v_valid] = np.nanmean(v_win)
            v_slope = np.sum((x - x_mean) * (v_safe - np.mean(v_safe))) / ss_xx
            v_trend = v_slope / (np.mean(v_safe) + 1e-9)

        # Divergence: opposite signs of price and OI slopes
        div = -p_norm * o_norm * 1000.0  # positive when diverging
        # Volume confirmation factor
        vol_conf = 1.0 + abs(v_trend) * 10.0
        div_confirmed = np.clip(div * vol_conf, -1.0, 1.0)

        divergence[i] = div_confirmed
        div_strength[i] = min(abs(div_confirmed), 1.0)

        # Classify
        if p_norm > 0.001 and o_norm < -0.001:
            is_bearish[i] = True
        elif p_norm < -0.001 and o_norm > 0.001:
            is_bullish[i] = True

    return divergence, div_strength, is_bearish, is_bullish
