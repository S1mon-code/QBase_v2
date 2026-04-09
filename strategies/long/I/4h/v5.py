"""MildTrendLongI4hV5 — LinearReg(60) + Fisher(30) + OBV(50).

Economic logic: Linear regression end-point provides a statistical trend anchor for
4H iron ore. Fisher Transform normalizes price into Gaussian distribution — positive
values confirm bullish state. Rising OBV validates volume participation. Signal scales
with Fisher magnitude and OBV trend.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.fisher_transform import fisher_transform
from indicators.trend.linear_regression import linear_regression
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV5(TrendingStrategy):
    name = "long_I_4h_v5"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 70

    lr_period: int = 60
    fisher_period: int = 30
    obv_smooth: int = 50
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr = linear_regression(self._closes, period=self.lr_period)
        self._fisher, self._fisher_trigger = fisher_transform(
            self._highs, self._lows, period=self.fisher_period
        )
        self._obv_raw = obv(self._closes, self._volumes)
        n = len(self._closes)
        self._obv_ma = np.full(n, np.nan)
        for i in range(self.obv_smooth - 1, n):
            self._obv_ma[i] = np.mean(self._obv_raw[i - self.obv_smooth + 1:i + 1])

    def _generate_signal(self, bar_index: int) -> float:
        lr_val = self._lr[bar_index]
        f_val = self._fisher[bar_index]
        obv_now = self._obv_raw[bar_index]
        obv_ma = self._obv_ma[bar_index]
        close = self._closes[bar_index]

        if any(np.isnan(v) for v in [lr_val, f_val, obv_now, obv_ma, close]):
            return 0.0

        above_lr = close > lr_val
        fisher_bullish = f_val > 0.0
        obv_rising = obv_now > obv_ma

        if not (above_lr and fisher_bullish and obv_rising):
            return 0.0

        fisher_score = min(1.0, f_val / 2.0) * 0.4
        return min(1.0, 0.3 + fisher_score + 0.2)

    def get_indicator_config(self):
        return [
            {"name": f"LinReg({self.lr_period})", "array": self._lr, "type": "overlay"},
            {"name": "Fisher", "array": self._fisher, "type": "subplot", "panel": "Fisher", "zero_line": True},
            {"name": "Trigger", "array": self._fisher_trigger, "type": "subplot", "panel": "Fisher", "style": "dash"},
            {"name": "OBV", "array": self._obv_raw, "type": "subplot", "panel": "OBV"},
            {"name": "OBV MA", "array": self._obv_ma, "type": "subplot", "panel": "OBV", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"LinReg({self.lr_period})", datetimes, self._lr, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Fisher Transform",
                    [
                        self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#bb86fc"),
                        self._make_subplot_trace("Trigger", datetimes, self._fisher_trigger, color="#ff8a80", style="dash"),
                    ],
                    zero_line=True,
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
