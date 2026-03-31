import numpy as np


def smart_money_index(opens: np.ndarray, closes: np.ndarray,
                      highs: np.ndarray, lows: np.ndarray,
                      volumes: np.ndarray, period: int = 20) -> tuple:
    """Smart Money Index approximated from daily OHLC.

    Theory: first 30min is dumb money (emotional, retail), last 30min is
    smart money (institutional). Approximated from daily data as:
    - Dumb money move = open direction (gap from prev close)
    - Smart money move = close direction (intraday close vs open)

    SMI = cumulative(smart_move - dumb_move)

    Parameters
    ----------
    opens, closes, highs, lows : np.ndarray
        OHLC prices.
    volumes : np.ndarray
        Trading volumes.
    period : int
        Smoothing period for signal line.

    Returns
    -------
    smi : np.ndarray
        Smart Money Index (cumulative).
    smi_signal : np.ndarray
        SMA-smoothed SMI signal line.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    smi = np.full(n, np.nan)
    smi_signal = np.full(n, np.nan)

    if n < 2:
        return smi, smi_signal

    # Smart move: close - open (intraday institutional flow)
    smart_move = closes - opens

    # Dumb move: open - prev_close (overnight gap, retail reaction)
    dumb_move = np.empty(n)
    dumb_move[0] = 0.0
    dumb_move[1:] = opens[1:] - closes[:-1]

    # Normalize by range to make comparable across price levels
    daily_range = highs - lows
    daily_range = np.where(daily_range > 0, daily_range, 1.0)

    smart_norm = smart_move / daily_range
    dumb_norm = dumb_move / daily_range

    # Cumulative SMI
    diff = smart_norm - dumb_norm
    diff[0] = 0.0
    smi[0] = 0.0
    for i in range(1, n):
        if np.isfinite(diff[i]):
            smi[i] = smi[i - 1] + diff[i]
        else:
            smi[i] = smi[i - 1]

    # Signal line: SMA of SMI
    for i in range(period - 1, n):
        window = smi[i - period + 1:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) > 0:
            smi_signal[i] = np.mean(valid)

    return smi, smi_signal
