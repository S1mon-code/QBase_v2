import numpy as np


def trend_strength(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Composite trend strength from multiple metrics.

    Blends three components into a 0-100 score:
    1. Efficiency ratio (net move / total path)
    2. R-squared of linear regression on closes
    3. Directional movement ratio (+DM vs -DM)

    Returns trend_strength_score (0-100).
    """
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return out

    x = np.arange(period, dtype=np.float64)
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean) ** 2)

    for i in range(period, n):
        # --- Component 1: Efficiency Ratio (0-1) ---
        net_move = abs(closes[i] - closes[i - period])
        total_move = np.sum(np.abs(np.diff(closes[i - period : i + 1])))
        er = net_move / (total_move + 1e-14)

        # --- Component 2: R-squared of linear regression ---
        y = closes[i - period : i]
        if len(y) != period:
            continue
        y_mean = np.mean(y)
        ss_y = np.sum((y - y_mean) ** 2)
        if ss_y < 1e-14:
            r2 = 0.0
        else:
            ss_xy = np.sum((x - x_mean) * (y - y_mean))
            r2 = (ss_xy ** 2) / (ss_x * ss_y)
            r2 = float(np.clip(r2, 0.0, 1.0))

        # --- Component 3: Directional Movement Ratio ---
        plus_dm_sum = 0.0
        minus_dm_sum = 0.0
        for j in range(i - period + 1, i + 1):
            up_move = highs[j] - highs[j - 1]
            down_move = lows[j - 1] - lows[j]

            if up_move > down_move and up_move > 0:
                plus_dm_sum += up_move
            if down_move > up_move and down_move > 0:
                minus_dm_sum += down_move

        dm_total = plus_dm_sum + minus_dm_sum
        if dm_total > 1e-14:
            dm_ratio = abs(plus_dm_sum - minus_dm_sum) / dm_total
        else:
            dm_ratio = 0.0

        # --- Combine: weighted average, scale to 0-100 ---
        score = (0.40 * er + 0.35 * r2 + 0.25 * dm_ratio) * 100.0
        out[i] = float(np.clip(score, 0.0, 100.0))

    return out
