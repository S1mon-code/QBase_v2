"""Regime-adaptive Chandelier Exit trailing stop.

Long:  stop = highest_since_entry - atr_mult * ATR(14)
Short: stop = lowest_since_entry  + atr_mult * ATR(14)

ATR multiplier depends on regime:
  strong_trend:   2.5-3.5 (wide, let trend breathe)
  mild_trend:     2.0-2.5
  mean_reversion: 1.5-2.0 (from entry price, not extremum)
  crisis:         1.5 fixed + time stop
"""

from __future__ import annotations

import numpy as np


# Default ATR multiplier ranges per regime (midpoint used when not specified).
_REGIME_DEFAULTS: dict[str, float] = {
    "strong_trend": 3.0,
    "mild_trend": 2.25,
    "mean_reversion": 1.75,
    "crisis": 1.5,
    # Backward-compat alias
    "trending": 3.0,
}

# Crisis time-stop default: bars without profit before forced exit.
_CRISIS_TIME_STOP_BARS: int = 10


class ChandelierExit:
    """Regime-adaptive trailing stop using the Chandelier Exit method.

    Parameters
    ----------
    atr_mult : float
        ATR multiplier for stop distance.  When *None* the class picks a
        default based on ``regime``.
    regime : str
        One of ``strong_trend``, ``mild_trend``, ``mean_reversion``,
        ``crisis``, or the alias ``trending``.
    crisis_time_stop : int
        Number of bars without profit before forced exit in crisis regime.
    """

    def __init__(
        self,
        atr_mult: float | None = None,
        regime: str = "trending",
        crisis_time_stop: int = _CRISIS_TIME_STOP_BARS,
    ) -> None:
        """Initialise the Chandelier Exit."""
        self._regime = regime
        self._atr_mult = atr_mult if atr_mult is not None else _REGIME_DEFAULTS.get(regime, 3.0)
        self._crisis_time_stop = crisis_time_stop

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

        if self._regime == "mean_reversion":
            # Mean-reversion: stop is measured from *entry price*, not extremum.
            if side == 1:
                new_stop = self._entry_price - distance
            else:
                new_stop = self._entry_price + distance
        else:
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
        """Return True if the current close has hit the stop.

        Also triggers in crisis regime when the time-stop condition is met
        (held for ``crisis_time_stop`` bars without any profit).
        """
        if np.isnan(self._stop) or side == 0:
            return False

        # Price stop.
        if side == 1 and close <= self._stop:
            return True
        if side == -1 and close >= self._stop:
            return True

        # Crisis time stop: N bars without any profit.
        if self._regime == "crisis":
            if self._bars_since_entry >= self._crisis_time_stop and self._best_pnl <= 0.0:
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
            Regime label (affects mean_reversion logic).

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

        use_entry = regime == "mean_reversion"

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

            if use_entry:
                entry_p = entries[i]
                raw = (entry_p - dist) if s == 1 else (entry_p + dist)
            else:
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
