import numpy as np


def oi_accumulation(closes: np.ndarray, oi: np.ndarray,
                    period: int = 20) -> tuple:
    """OI accumulation/distribution indicator.

    Tracks whether OI is building on up-moves (bullish accumulation)
    or on down-moves (bearish accumulation).  Conceptually similar
    to A/D line but using OI instead of volume.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    period : int
        Smoothing period for the signal line (SMA).

    Returns
    -------
    oi_ad : np.ndarray
        Cumulative OI accumulation/distribution line.
    oi_ad_signal : np.ndarray
        SMA of oi_ad (signal line).
    """
    n = len(closes)
    oi_ad = np.full(n, np.nan)
    oi_ad_signal = np.full(n, np.nan)

    if n < 2:
        return oi_ad, oi_ad_signal

    oi_ad[0] = 0.0
    for i in range(1, n):
        price_chg = closes[i] - closes[i - 1]
        oi_chg = oi[i] - oi[i - 1]

        # OI builds on up-moves → positive contribution
        # OI builds on down-moves → negative contribution
        if price_chg > 0 and oi_chg > 0:
            contrib = oi_chg  # bullish accumulation
        elif price_chg < 0 and oi_chg > 0:
            contrib = -oi_chg  # bearish accumulation
        elif price_chg > 0 and oi_chg < 0:
            contrib = oi_chg  # bullish but OI declining (short covering)
        elif price_chg < 0 and oi_chg < 0:
            contrib = -oi_chg  # long liquidation (positive = relief)
        else:
            contrib = 0.0

        oi_ad[i] = oi_ad[i - 1] + contrib

    # Signal line (SMA)
    if n >= period:
        for i in range(period - 1, n):
            oi_ad_signal[i] = np.mean(oi_ad[i - period + 1:i + 1])

    return oi_ad, oi_ad_signal
