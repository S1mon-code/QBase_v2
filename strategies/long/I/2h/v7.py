"""MildTrendLongI2hV7 — ZLEMA(45) + Schaff(60,30,15) + OBV(60).

Economic logic: ZLEMA reduces inherent EMA lag for responsive 2H iron ore tracking.
Schaff Trend Cycle confirms momentum via double-stochastic MACD smoothing.
Rising OBV validates volume participation in the trend. Signal scales with STC level
and OBV trend strength.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.trend.zlema import zlema
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV7(TrendingStrategy):
    name = "long_I_2h_v7"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 70

    zlema_period: int = 45
    schaff_period: int = 60
    obv_smooth: int = 60
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._zlema = zlema(self._closes, period=self.zlema_period)
        self._schaff = schaff_trend_cycle(self._closes, period=self.schaff_period, fast=30, slow=15)
        self._obv_raw = obv(self._closes, self._volumes)
        n = len(self._closes)
        self._obv_ma = np.full(n, np.nan)
        for i in range(self.obv_smooth - 1, n):
            self._obv_ma[i] = np.mean(self._obv_raw[i - self.obv_smooth + 1:i + 1])

    def _generate_signal(self, bar_index: int) -> float:
        z = self._zlema[bar_index]
        z_prev = self._zlema[bar_index - 1] if bar_index > 0 else np.nan
        stc = self._schaff[bar_index]
        obv_now = self._obv_raw[bar_index]
        obv_ma = self._obv_ma[bar_index]

        if any(np.isnan(v) for v in [z, z_prev, stc, obv_now, obv_ma]):
            return 0.0

        zlema_rising = z > z_prev
        schaff_bullish = stc > 25.0
        obv_rising = obv_now > obv_ma

        if not (zlema_rising and schaff_bullish and obv_rising):
            return 0.0

        stc_score = min(1.0, (stc - 25.0) / 75.0) * 0.5
        return min(1.0, 0.3 + stc_score + 0.2)

    def get_indicator_config(self):
        return [
            {"name": f"ZLEMA({self.zlema_period})", "array": self._zlema, "type": "overlay"},
            {"name": f"Schaff({self.schaff_period})", "array": self._schaff, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [25, 75]},
            {"name": "OBV", "array": self._obv_raw, "type": "subplot", "panel": "OBV"},
            {"name": "OBV MA", "array": self._obv_ma, "type": "subplot", "panel": "OBV", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"ZLEMA({self.zlema_period})", datetimes, self._zlema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Schaff Trend Cycle",
                    [self._make_subplot_trace("STC", datetimes, self._schaff, color="#bb86fc")],
                    y_range=[0, 100], horizontal_lines=[25, 75],
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv_raw, color="#66bb6a"),
                        self._make_subplot_trace("OBV MA", datetimes, self._obv_ma, color="#ef5350", style="dash"),
                    ],
                ),
            ],
        }
