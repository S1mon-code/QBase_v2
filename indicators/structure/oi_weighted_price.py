import numpy as np


def oi_weighted_price(closes: np.ndarray, oi: np.ndarray,
                      period: int = 20) -> tuple:
    """OI-weighted average price (like VWAP but weighted by OI).

    Price above the OI-WAP means positioned buyers are in profit
    on average.  Price below means positioned buyers are underwater.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    period : int
        Lookback window.

    Returns
    -------
    oi_wap : np.ndarray
        OI-weighted average price over the lookback.
    price_deviation : np.ndarray
        (close - oi_wap) / oi_wap.  Positive = price above WAP.
    """
    n = len(closes)
    oi_wap = np.full(n, np.nan)
    price_deviation = np.full(n, np.nan)

    if n < period:
        return oi_wap, price_deviation

    for i in range(period - 1, n):
        c_win = closes[i - period + 1:i + 1]
        o_win = oi[i - period + 1:i + 1]
        total_oi = np.sum(o_win)

        if total_oi > 0:
            wap = np.sum(c_win * o_win) / total_oi
            oi_wap[i] = wap
            price_deviation[i] = (closes[i] - wap) / wap if wap > 0 else 0.0
        else:
            oi_wap[i] = closes[i]
            price_deviation[i] = 0.0

    return oi_wap, price_deviation
