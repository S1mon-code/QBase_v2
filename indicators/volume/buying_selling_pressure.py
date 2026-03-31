import numpy as np


def buying_selling_pressure(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 14,
) -> tuple:
    """Decompose volume into buying pressure and selling pressure.

    Buying pressure = (Close - Low) / (High - Low) * Volume
    Selling pressure = (High - Close) / (High - Low) * Volume
    Pressure ratio = buy / sell. > 1 = buyers dominating.
    Returns (buy_pressure, sell_pressure, pressure_ratio) — all smoothed over period.
    """
    n = len(closes)
    if n == 0:
        emp = np.array([], dtype=np.float64)
        return emp.copy(), emp.copy(), emp.copy()

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    hl_range = highs - lows

    # Raw buying/selling pressure per bar
    raw_buy = np.zeros(n, dtype=np.float64)
    raw_sell = np.zeros(n, dtype=np.float64)
    mask = hl_range > 0
    raw_buy[mask] = (closes[mask] - lows[mask]) / hl_range[mask] * volumes[mask]
    raw_sell[mask] = (highs[mask] - closes[mask]) / hl_range[mask] * volumes[mask]

    # Smooth over period (SMA)
    buy_pressure = np.full(n, np.nan)
    sell_pressure = np.full(n, np.nan)
    pressure_ratio = np.full(n, np.nan)

    for i in range(period - 1, n):
        bp = np.mean(raw_buy[i - period + 1 : i + 1])
        sp = np.mean(raw_sell[i - period + 1 : i + 1])
        buy_pressure[i] = bp
        sell_pressure[i] = sp
        if sp > 0:
            pressure_ratio[i] = bp / sp

    return buy_pressure, sell_pressure, pressure_ratio
