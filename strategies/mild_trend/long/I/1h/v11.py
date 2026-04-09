"""MildTrendLongI1hV11 — EMA Cross + RSI Momentum.

Economic logic: Dual EMA crossover confirms trend direction while RSI above 50
validates bullish momentum. Signal strength scales with RSI distance from the
neutral zone, entering stronger positions when momentum is decisive and weaker
ones during tentative trend conditions. Tuned for 1H iron ore with 20-50 bar
lookbacks capturing 1-5 day trend moves.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV11(TrendingStrategy):
    """EMA(fast) > EMA(slow) with RSI > 50 momentum confirmation.

    Signal logic:
        ema_fast > ema_slow AND rsi > 50 -> signal = min(1.0, (rsi - 50) / 50 + 0.3)
        else -> 0.0
    """

    name = "mild_trend_long_I_1h_v11"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 60  # slow_period(50) + buffer(10)

    # Optimizable parameters
    fast_period: int = 20
    slow_period: int = 50
    rsi_period: int = 14
    chandelier_mult: float = 2.5

    def on_init_arrays(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        volumes: np.ndarray,
        oi: np.ndarray,
        datetimes: np.ndarray,
    ) -> None:
        """Precompute EMA and RSI arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = ema(self._closes, period=self.fast_period)
        self._ema_slow = ema(self._closes, period=self.slow_period)
        self._rsi = rsi(self._closes, period=self.rsi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal based on EMA crossover with RSI momentum filter."""
        ema_f = self._ema_fast[bar_index]
        ema_s = self._ema_slow[bar_index]
        rsi_val = self._rsi[bar_index]

        if np.isnan(ema_f) or np.isnan(ema_s) or np.isnan(rsi_val):
            return 0.0

        if ema_f > ema_s and rsi_val > 50.0:
            return min(1.0, (rsi_val - 50.0) / 50.0 + 0.3)

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"EMA({self.fast_period})", "array": self._ema_fast},
            {"name": f"EMA({self.slow_period})", "array": self._ema_slow},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"EMA({self.fast_period})", datetimes, self._ema_fast, color="#ffab40"),
            self._make_overlay(f"EMA({self.slow_period})", datetimes, self._ema_slow, color="#ab47bc")
        ]
        subplots = [
            self._make_subplot(
                f"RSI({self.rsi_period})",
                [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc")],
                horizontal_lines=[30, 70], y_range=[0, 100],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}

