"""StrongTrendLongAG1hV10 — LinearReg(40) + Aroon(25) + OBV(30).

Economic logic: Linear Regression captures Silver's 1H directional slope.
Aroon with short period detects whether AG is making new highs consistently.
OBV trend confirms cumulative buying volume pressure.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression
from indicators.trend.aroon import aroon
from indicators.volume.obv import obv
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG1hV10(TrendingStrategy):
    name = "strong_trend_long_AG_1h_v10"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    linreg_period: int = 40
    aroon_period: int = 25
    obv_ema_period: int = 30
    chandelier_mult: float = 2.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._linreg = linear_regression(self._closes, period=self.linreg_period)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period,
        )
        raw_obv = obv(self._closes, self._volumes)
        self._obv_ema = ema(raw_obv, period=self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        lr = self._linreg[bar_index]
        lr_prev = self._linreg[bar_index - 1]
        ao = self._aroon_osc[bar_index]
        oe = self._obv_ema[bar_index]
        oe_prev = self._obv_ema[bar_index - 1]

        if any(np.isnan(v) for v in (lr, lr_prev, ao, oe, oe_prev)):
            return 0.0

        if lr > lr_prev and ao > 0.0 and oe > oe_prev:
            strength = min(1.0, ao / 80.0 * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"LinReg({self.linreg_period})", "array": self._linreg, "type": "overlay"},
            {"name": f"Aroon Osc({self.aroon_period})", "array": self._aroon_osc,
             "type": "subplot", "zero_line": True, "y_range": [-100, 100]},
            {"name": f"OBV EMA({self.obv_ema_period})", "array": self._obv_ema, "type": "subplot"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"LinReg({self.linreg_period})", datetimes, self._linreg, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Aroon({self.aroon_period})",
                    [self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#66bb6a"),
                     self._make_subplot_trace("Down", datetimes, self._aroon_down, color="#ef5350")],
                    y_range=[0, 100],
                ),
                self._make_subplot(
                    f"OBV EMA({self.obv_ema_period})",
                    [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#42a5f5")],
                ),
            ],
        }
