import numpy as np

from indicators._utils import _rsi, _sma


def stoch_rsi(
    closes: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic RSI (Tushar Chande & Stanley Kroll).

    StochRSI = (RSI - RSI_low) / (RSI_high - RSI_low) over stoch_period
    %K = SMA(StochRSI, k_period)
    %D = SMA(%K, d_period)

    Range [0, 1] (or [0, 100] when multiplied by 100).
    Here we return values in [0, 1] scale.
    Returns (%K, %D).
    """
    n = closes.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty
    nans = np.full(n, np.nan)
    if n <= rsi_period + stoch_period:
        return nans.copy(), nans.copy()

    rsi_values = _rsi(closes, rsi_period)

    # Stochastic of RSI
    stoch_raw = np.full(n, np.nan)
    for i in range(rsi_period + stoch_period - 1, n):
        window = rsi_values[i - stoch_period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        lo = np.min(window)
        hi = np.max(window)
        if hi == lo:
            stoch_raw[i] = 0.5
        else:
            stoch_raw[i] = (rsi_values[i] - lo) / (hi - lo)

    # %K = SMA of stoch_raw
    k_line = _sma(stoch_raw, k_period)

    # %D = SMA of %K
    d_line = _sma(k_line, d_period)

    return k_line, d_line
