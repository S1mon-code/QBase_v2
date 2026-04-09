"""Regime-adaptive Chandelier Exit trailing stop.

Long:  stop = highest_since_entry - atr_mult * ATR(14)
Short: stop = lowest_since_entry  + atr_mult * ATR(14)

ATR multiplier defaults:
  long:      2.5
  short:     2.5
  trending:  2.5 (backward-compat alias)
"""

from __future__ import annotations

import numpy as np


# Default ATR multiplier per regime.
_REGIME_DEFAULTS: dict[str, float] = {
    "long": 2.5,
    "short": 2.5,
    # Backward-compat aliases
    "trending": 2.5,
    "long": 2.5,
    "short": 2.5,
    "mean_reversion": 2.5,
    "crisis": 2.5,
}


class ChandelierExit:
    """Regime-adaptive trailing stop using the Chandelier Exit method.

    Parameters
    ----------
    atr_mult : float
        ATR multiplier for stop distance.  When *None* the class picks a
        default based on ``regime``.
    regime : str
        One of ``long``, ``short``, or the alias ``trending``.
    """

    def __init__(
        self,
        atr_mult: float | None = None,
        regime: str = "trending",
    ) -> None:
        """Initialise the Chandelier Exit."""
        self._regime = regime
        self._atr_mult = atr_mult if atr_mult is not None else _REGIME_DEFAULTS.get(regime, 2.5)

        # Internal mutable state – reset per trade.
        self._highest: float = -np.inf
        self._lowest: float = np.inf
        self._entry_price: float = np.nan
        self._stop: float = np.nan
        self._bars_since_entry: int = 0
        self._best_pnl: float = 0.0
        self._side: int = 0  # +1 long, -1 short

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        high: float,
        low: float,
        close: float,
        atr: float,
        side: int,
    ) -> None:
        """Update internal state with a new bar.

        Parameters
        ----------
        high, low, close : float
            Bar OHLC (only H/L/C needed).
        atr : float
            Current ATR value.
        side : int
            +1 for long, -1 for short, 0 for flat.
        """
        if side == 0:
            return

        # First bar of a new trade – record entry.
        if self._side == 0:
            self._entry_price = close
            self._highest = high
            self._lowest = low
            self._side = side

        self._bars_since_entry += 1
        self._highest = max(self._highest, high)
        self._lowest = min(self._lowest, low)

        distance = self._atr_mult * atr

        if side == 1:
            new_stop = self._highest - distance
        else:
            new_stop = self._lowest + distance

        # Ratchet: long stop can only rise; short stop can only fall.
        if np.isnan(self._stop):
            self._stop = new_stop
        elif side == 1:
            self._stop = max(self._stop, new_stop)
        else:
            self._stop = min(self._stop, new_stop)

        # Track best unrealised PnL (for crisis time stop).
        pnl = (close - self._entry_price) * side
        self._best_pnl = max(self._best_pnl, pnl)

    def get_stop(self) -> float:
        """Return the current stop level."""
        return self._stop

    def is_stopped(self, close: float, side: int) -> bool:
        """Return True if the current close has hit the stop."""
        if np.isnan(self._stop) or side == 0:
            return False

        # Price stop.
        if side == 1 and close <= self._stop:
            return True
        if side == -1 and close >= self._stop:
            return True

        return False

    def reset(self) -> None:
        """Reset all state for a new trade."""
        self._highest = -np.inf
        self._lowest = np.inf
        self._entry_price = np.nan
        self._stop = np.nan
        self._bars_since_entry = 0
        self._best_pnl = 0.0
        self._side = 0

    # ------------------------------------------------------------------
    # Vectorised / precomputed mode
    # ------------------------------------------------------------------

    @staticmethod
    def compute_stops(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atrs: np.ndarray,
        entries: np.ndarray,
        sides: np.ndarray,
        atr_mult: float,
        regime: str = "trending",
    ) -> np.ndarray:
        """Vectorised stop computation for precomputed (``on_init_arrays``) mode.

        Parameters
        ----------
        highs, lows, closes, atrs : np.ndarray
            Price and ATR arrays of equal length.
        entries : np.ndarray
            Entry price at each bar (NaN when flat).
        sides : np.ndarray
            +1 long, -1 short, 0 flat at each bar.
        atr_mult : float
            ATR multiplier.
        regime : str
            Regime label (``"long"`` or ``"short"``).

        Returns
        -------
        np.ndarray
            Stop level at each bar (NaN when flat).
        """
        n = len(closes)
        stops = np.full(n, np.nan)
        highest = -np.inf
        lowest = np.inf
        prev_side = 0
        stop = np.nan

        for i in range(n):
            s = int(sides[i])
            if s == 0:
                highest = -np.inf
                lowest = np.inf
                stop = np.nan
                prev_side = 0
                continue

            # New trade detection.
            if prev_side == 0:
                highest = highs[i]
                lowest = lows[i]
                stop = np.nan

            highest = max(highest, highs[i])
            lowest = min(lowest, lows[i])

            dist = atr_mult * atrs[i]

            raw = (highest - dist) if s == 1 else (lowest + dist)

            if np.isnan(stop):
                stop = raw
            elif s == 1:
                stop = max(stop, raw)
            else:
                stop = min(stop, raw)

            stops[i] = stop
            prev_side = s

        return stops
