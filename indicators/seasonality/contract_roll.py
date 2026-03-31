"""Contract roll effect — measures return and volatility patterns during futures rollover windows."""
import numpy as np


def roll_effect(closes: np.ndarray, is_rollover: np.ndarray,
                period: int = 20) -> tuple:
    """Contract roll period behavior analysis.

    Measures return and volatility patterns during contract rollover windows.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    is_rollover : np.ndarray
        Boolean array, True on rollover/expiry transition bars.
    period : int
        Window around rollover dates to analyze.

    Returns
    -------
    roll_return : np.ndarray
        Average return during roll windows (rolling computation).
    roll_vol : np.ndarray
        Volatility during roll windows relative to normal periods.
    in_roll_window : np.ndarray
        Boolean array, True if within `period` bars of a rollover.
    """
    n = len(closes)
    if n == 0:
        return (np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=bool))

    roll_return = np.full(n, np.nan)
    roll_vol = np.full(n, np.nan)
    in_roll_window = np.zeros(n, dtype=bool)

    is_roll = is_rollover.astype(bool)

    # Mark bars within `period` bars of a rollover
    roll_indices = np.where(is_roll)[0]
    for idx in roll_indices:
        lo = max(0, idx - period // 2)
        hi = min(n, idx + period // 2 + 1)
        in_roll_window[lo:hi] = True

    # Daily returns
    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    # Rolling statistics
    lookback = max(period * 5, 100)
    for i in range(lookback, n):
        start = i - lookback
        window_rets = rets[start:i + 1]
        window_roll = in_roll_window[start:i + 1]
        valid = np.isfinite(window_rets)

        roll_mask = window_roll & valid
        non_roll_mask = (~window_roll) & valid

        roll_r = window_rets[roll_mask]
        non_roll_r = window_rets[non_roll_mask]

        if len(roll_r) >= 3:
            roll_return[i] = np.mean(roll_r)

            roll_std = np.std(roll_r)
            non_roll_std = np.std(non_roll_r) if len(non_roll_r) >= 3 else roll_std
            if non_roll_std > 0:
                roll_vol[i] = roll_std / non_roll_std
            else:
                roll_vol[i] = 1.0

    return roll_return, roll_vol, in_roll_window
