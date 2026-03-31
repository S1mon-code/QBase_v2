import numpy as np

from indicators._utils import _rsi


def connors_rsi(
    closes: np.ndarray,
    rsi_period: int = 3,
    streak_period: int = 2,
    pct_rank_period: int = 100,
) -> np.ndarray:
    """Connors RSI.

    CRSI = (RSI(close, rsi_period) + RSI(streak, streak_period) + PercentRank) / 3

    Components:
    1. Short-term RSI of price
    2. RSI of consecutive up/down streak length
    3. Percent rank of current price change over lookback window

    Range [0, 100].
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n <= max(rsi_period, streak_period) + 1:
        return np.full(n, np.nan)

    # Component 1: RSI of price
    rsi_price = _rsi(closes, rsi_period)

    # Component 2: Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif closes[i] < closes[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
        else:
            streak[i] = 0

    rsi_streak = _rsi(streak, streak_period)

    # Component 3: Percent rank of today's price change
    pct_changes = np.diff(closes)
    pct_rank = np.full(n, np.nan)
    for i in range(pct_rank_period, len(pct_changes)):
        current = pct_changes[i]
        lookback = pct_changes[i - pct_rank_period:i]
        pct_rank[i + 1] = 100.0 * np.sum(lookback < current) / pct_rank_period

    # Composite — average available components, NaN where none are valid
    components = np.array([rsi_price, rsi_streak, pct_rank])
    valid_count = np.sum(~np.isnan(components), axis=0)
    comp_sum = np.nansum(components, axis=0)
    safe_count = np.where(valid_count > 0, valid_count, 1)
    result = np.where(valid_count > 0, comp_sum / safe_count, np.nan)

    return result
