"""StrongTrendShortAGDailyV4 — FRAMA Bearish + TSI Negative + RSI Weak.

Economic logic: FRAMA (fractal adaptive MA) adapts to silver's fractal nature.
Price below FRAMA confirms bearish trend. TSI turning negative shows momentum loss.
RSI below 45 confirms the asset is in bearish territory.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.momentum.tsi import tsi
from indicators.trend.frama import frama
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV4(TrendingStrategy):
    """FRAMA bearish + TSI negative + RSI weak.

    Signal logic:
        close < FRAMA AND TSI < 0 AND RSI < 40 -> -0.85
        close < FRAMA AND TSI < 0 -> -0.50
        else -> 0.0
    """

    name = "short_AG_daily_v4"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 200

    frama_period: int = 120
    tsi_long: int = 60
    rsi_period: int = 80
    chandelier_mult: float = 4.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._frama = frama(self._closes, period=self.frama_period)
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=30,
        )
        self._rsi = rsi(self._closes, period=self.rsi_period)

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        f = self._frama[bar_index]
        t = self._tsi_line[bar_index]
        r = self._rsi[bar_index]

        if np.isnan(c) or np.isnan(f) or np.isnan(t):
            return 0.0

        if c >= f:
            return 0.0

        if t < 0 and (not np.isnan(r) and r < 40):
            return -0.85
        if t < 0:
            return -0.50
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"FRAMA({self.frama_period})", "array": self._frama, "type": "overlay"},
            {"name": "TSI", "array": self._tsi_line, "panel": "TSI", "zero_line": True},
            {"name": "TSI Signal", "array": self._tsi_signal, "panel": "TSI"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi,
             "y_range": [0, 100], "horizontal_lines": [40, 60]},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"FRAMA({self.frama_period})", datetimes, self._frama, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot("TSI", [
                self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#42a5f5"),
                self._make_subplot_trace("TSI Signal", datetimes, self._tsi_signal, color="#ff7043"),
            ], zero_line=True),
            self._make_subplot(f"RSI({self.rsi_period})", [
                self._make_subplot_trace("RSI", datetimes, self._rsi, color="#ab47bc"),
            ], horizontal_lines=[40, 60], y_range=[0, 100]),
        ]
        return {"overlays": overlays, "subplots": subplots}
