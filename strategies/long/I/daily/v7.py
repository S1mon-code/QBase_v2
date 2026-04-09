"""MildTrendLongIDailyV7 — FRAMA(120) + MACD(60,130,45) + BSP(80).

Economic logic: FRAMA provides fractal-adaptive trend filtering for iron ore's
regime shifts. Wide MACD parameters (60/130) capture structural momentum moves.
Buying/Selling Pressure ratio confirms volume-side conviction with buyers dominating.
Signal strength scales with MACD histogram magnitude and pressure ratio.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.frama import frama
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV7(TrendingStrategy):
    name = "long_I_daily_v7"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 160

    frama_period: int = 120
    macd_fast: int = 60
    macd_slow: int = 130
    bsp_period: int = 80
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._frama = frama(self._closes, period=self.frama_period)
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=45
        )
        self._buy_p, self._sell_p, self._pressure_ratio = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes, period=self.bsp_period
        )

    def _generate_signal(self, bar_index: int) -> float:
        fr = self._frama[bar_index]
        mh = self._macd_hist[bar_index]
        pr = self._pressure_ratio[bar_index]
        close = self._closes[bar_index]

        if any(np.isnan(v) for v in [fr, mh, pr, close]):
            return 0.0

        above_frama = close > fr
        macd_bullish = mh > 0.0
        buying_dominant = pr > 1.0

        if not (above_frama and macd_bullish and buying_dominant):
            return 0.0

        # Scale with MACD hist and pressure ratio
        macd_score = min(1.0, mh / (close * 0.02)) * 0.4 if close > 0 else 0.0
        pr_score = min(1.0, (pr - 1.0) / 0.5) * 0.3
        return min(1.0, 0.3 + macd_score + pr_score)

    def get_indicator_config(self):
        return [
            {"name": f"FRAMA({self.frama_period})", "array": self._frama, "type": "overlay"},
            {"name": "MACD Line", "array": self._macd_line, "type": "subplot", "panel": "MACD", "zero_line": True},
            {"name": "MACD Signal", "array": self._macd_signal, "type": "subplot", "panel": "MACD", "style": "dash"},
            {"name": "MACD Hist", "array": self._macd_hist, "type": "subplot", "panel": "MACD", "style": "bar",
             "color_positive": "#66bb6a", "color_negative": "#ef5350"},
            {"name": "Pressure Ratio", "array": self._pressure_ratio, "type": "subplot", "horizontal_lines": [1.0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"FRAMA({self.frama_period})", datetimes, self._frama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "MACD",
                    [
                        self._make_subplot_trace("Line", datetimes, self._macd_line, color="#42a5f5"),
                        self._make_subplot_trace("Signal", datetimes, self._macd_signal, color="#ff8a80", style="dash"),
                        self._make_subplot_trace("Hist", datetimes, self._macd_hist, style="bar",
                                                 color_positive="#66bb6a", color_negative="#ef5350"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Buy/Sell Pressure",
                    [self._make_subplot_trace("Ratio", datetimes, self._pressure_ratio, color="#ab47bc")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
