"""Continuous volatility scaling.

position_scale = target_vol / realized_vol

Realized vol uses exponentially weighted moving standard deviation.
An additional extreme-vol layer further scales down when ATR percentile
is elevated.
"""

from __future__ import annotations

import numpy as np

from config import get_settings

# Annualisation factor (sqrt of trading days).
_SQRT_252 = np.sqrt(252)


def realized_vol(returns: np.ndarray, halflife: int = 60) -> np.ndarray:
    """Exponentially weighted annualised realised volatility.

    Parameters
    ----------
    returns : np.ndarray
        Array of simple returns.
    halflife : int
        Half-life in bars for the EW weighting (default 60).

    Returns
    -------
    np.ndarray
        Annualised realised volatility at each bar.
    """
    n = len(returns)
    result = np.full(n, np.nan)
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)

    ew_mean = 0.0
    ew_var = 0.0

    for i in range(n):
        r = returns[i]
        if i == 0:
            ew_mean = r
            ew_var = 0.0
        else:
            delta = r - ew_mean
            ew_mean = ew_mean + alpha * delta
            ew_var = (1.0 - alpha) * (ew_var + alpha * delta * delta)

        result[i] = np.sqrt(ew_var) * _SQRT_252

    return result


def vol_scale(target_vol: float, realized_vol_arr: np.ndarray) -> np.ndarray:
    """Compute position scale = target_vol / realized_vol, clipped to [0.2, 3.0].

    Parameters
    ----------
    target_vol : float
        Target annualised volatility (e.g. 0.10 for 10%).
    realized_vol_arr : np.ndarray
        Annualised realised vol array.

    Returns
    -------
    np.ndarray
        Scaling factor array.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(realized_vol_arr > 0, target_vol / realized_vol_arr, 1.0)
    return np.clip(raw, 0.2, 3.0)


def atr_percentile(atr: np.ndarray, lookback: int = 252) -> np.ndarray:
    """Rolling percentile rank of ATR over *lookback* bars.

    Parameters
    ----------
    atr : np.ndarray
        ATR values.
    lookback : int
        Rolling window size (default 252).

    Returns
    -------
    np.ndarray
        Percentile rank in [0, 100] at each bar.
    """
    n = len(atr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - lookback + 1)
        window = atr[start : i + 1]
        if len(window) < 2:
            result[i] = 50.0
            continue
        sorted_window = np.sort(window)
        idx = np.searchsorted(sorted_window, atr[i], side="right")
        result[i] = (idx / len(window)) * 100.0

    return result


def extreme_vol_adjustment(atr_pctl: np.ndarray) -> np.ndarray:
    """Extra scaling for extreme volatility.

    >90th percentile: multiply by 0.5
    >80th percentile: multiply by 0.75
    Otherwise: 1.0

    Parameters
    ----------
    atr_pctl : np.ndarray
        ATR percentile ranks (0-100).

    Returns
    -------
    np.ndarray
        Adjustment factors.
    """
    adj = np.ones_like(atr_pctl, dtype=float)
    adj = np.where(atr_pctl > 90, 0.5, adj)
    # 80 < pctl <= 90
    adj = np.where((atr_pctl > 80) & (atr_pctl <= 90), 0.75, adj)
    return adj


class VolTargeting:
    """Continuous volatility scaling wrapper.

    Combines ``vol_scale`` and ``extreme_vol_adjustment`` into a single
    callable that returns the final position multiplier array.
    """

    def __init__(
        self,
        target_vol: float | None = None,
        halflife: int | None = None,
        atr_lookback: int = 252,
    ) -> None:
        """Initialise from config or explicit values.

        Parameters
        ----------
        target_vol : float or None
            If *None*, read from ``config/settings.yaml``.
        halflife : int or None
            If *None*, read from ``config/settings.yaml``.
        atr_lookback : int
            Rolling window for ATR percentile.
        """
        settings = get_settings()
        self._target_vol = target_vol if target_vol is not None else settings["target_vol"]
        self._halflife = halflife if halflife is not None else settings["vol_halflife"]
        self._atr_lookback = atr_lookback

    def compute(
        self,
        returns: np.ndarray,
        atr: np.ndarray,
    ) -> np.ndarray:
        """Return final position multiplier array.

        Parameters
        ----------
        returns : np.ndarray
            Simple return series.
        atr : np.ndarray
            ATR series (same length).

        Returns
        -------
        np.ndarray
            Combined scale factor at each bar.
        """
        rvol = realized_vol(returns, self._halflife)
        base_scale = vol_scale(self._target_vol, rvol)
        pctl = atr_percentile(atr, self._atr_lookback)
        adj = extreme_vol_adjustment(pctl)
        return base_scale * adj
