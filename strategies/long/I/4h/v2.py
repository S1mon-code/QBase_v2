"""MildTrendLongI4hV2 — SuperTrend(40,2.8) + MACD(20,50,15) + CMF(40).

Economic logic: SuperTrend with moderate ATR multiplier tracks 4H iron ore trends.
MACD histogram confirms momentum acceleration. CMF validates volume-weighted buying
pressure. Signal blends MACD histogram magnitude and CMF level for gradual sizing.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.supertrend import supertrend
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV2(TrendingStrategy):
    name = "long_I_4h_v2"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 65

    st_period: int = 40
    st_mult: float = 2.8
    macd_fast: int = 20
    macd_slow: int = 50
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes, period=self.st_period, multiplier=self.st_mult
        )
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=15
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, period=40)

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        mh = self._macd_hist[bar_index]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in [st_d, mh, cmf_val]):
            return 0.0

        if not (st_d == 1.0 and mh > 0.0 and cmf_val > 0.0):
            return 0.0

        close = self._closes[bar_index]
        macd_score = min(1.0, mh / (close * 0.01)) * 0.4 if close > 0 else 0.0
        cmf_score = min(1.0, cmf_val / 0.25) * 0.3
        return min(1.0, 0.3 + macd_score + cmf_score)

    def get_indicator_config(self):
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st_line, "type": "overlay", "style": "step"},
            {"name": "MACD Hist", "array": self._macd_hist, "type": "subplot", "panel": "MACD",
             "style": "bar", "color_positive": "#66bb6a", "color_negative": "#ef5350", "zero_line": True},
            {"name": "CMF", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st_line, style="step", color="#ff7043"),
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
                    "CMF",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
