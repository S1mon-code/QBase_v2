"""MildTrendLongI1hV10 — LinearReg(40) + Aroon(25) + CMF(25).

Economic logic: Linear regression provides a statistical trend anchor for 1H iron ore.
Aroon Up above Down with Up > 50 confirms bullish trend maturity. CMF validates
volume-weighted buying pressure. Signal scales with Aroon differential and CMF
magnitude for gradual position sizing.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.trend.linear_regression import linear_regression
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV10(TrendingStrategy):
    name = "mild_trend_long_I_1h_v10"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 45

    lr_period: int = 40
    aroon_period: int = 25
    cmf_period: int = 25
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr = linear_regression(self._closes, period=self.lr_period)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        lr_val = self._lr[bar_index]
        ar_up = self._aroon_up[bar_index]
        ar_down = self._aroon_down[bar_index]
        cmf_val = self._cmf[bar_index]
        close = self._closes[bar_index]

        if any(np.isnan(v) for v in [lr_val, ar_up, ar_down, cmf_val, close]):
            return 0.0

        above_lr = close > lr_val
        aroon_bullish = ar_up > ar_down and ar_up > 50.0
        cmf_positive = cmf_val > 0.0

        if not (above_lr and aroon_bullish and cmf_positive):
            return 0.0

        aroon_score = min(1.0, (ar_up - 50.0) / 50.0) * 0.4
        cmf_score = min(1.0, cmf_val / 0.25) * 0.3
        return min(1.0, 0.3 + aroon_score + cmf_score)

    def get_indicator_config(self):
        return [
            {"name": f"LinReg({self.lr_period})", "array": self._lr, "type": "overlay"},
            {"name": "Aroon Up", "array": self._aroon_up, "type": "subplot", "panel": "Aroon"},
            {"name": "Aroon Down", "array": self._aroon_down, "type": "subplot", "panel": "Aroon"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"LinReg({self.lr_period})", datetimes, self._lr, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Aroon",
                    [
                        self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#66bb6a"),
                        self._make_subplot_trace("Down", datetimes, self._aroon_down, color="#ef5350"),
                    ],
                    y_range=[0, 100], horizontal_lines=[50],
                ),
                self._make_subplot(
                    "CMF",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
