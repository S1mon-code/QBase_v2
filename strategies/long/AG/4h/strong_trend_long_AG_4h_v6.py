"""StrongTrendLongAG4hV6 — LinearReg(60) + TSI(40,20) + OBV(50).

Economic logic: Linear Regression on 4H bars captures Silver's directional
slope cleanly. TSI double-smoothed momentum filters AG's 4H noise. OBV
trend confirms cumulative buying pressure.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression
from indicators.momentum.tsi import tsi
from indicators.volume.obv import obv
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG4hV6(TrendingStrategy):
    name = "long_AG_4h_v6"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 100

    linreg_period: int = 60
    tsi_long: int = 40
    tsi_short: int = 20
    obv_ema_period: int = 50
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._linreg = linear_regression(self._closes, period=self.linreg_period)
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=self.tsi_short,
        )
        raw_obv = obv(self._closes, self._volumes)
        self._obv_ema = ema(raw_obv, period=self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        lr = self._linreg[bar_index]
        lr_prev = self._linreg[bar_index - 1]
        t = self._tsi_line[bar_index]
        oe = self._obv_ema[bar_index]
        oe_prev = self._obv_ema[bar_index - 1]

        if any(np.isnan(v) for v in (lr, lr_prev, t, oe, oe_prev)):
            return 0.0

        if lr > lr_prev and t > 0.0 and oe > oe_prev:
            strength = min(1.0, t / 25.0 * 0.6 + 0.3)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"LinReg({self.linreg_period})", "array": self._linreg, "type": "overlay"},
            {"name": f"TSI({self.tsi_long},{self.tsi_short})", "array": self._tsi_line,
             "type": "subplot", "zero_line": True},
            {"name": f"OBV EMA({self.obv_ema_period})", "array": self._obv_ema, "type": "subplot"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"LinReg({self.linreg_period})", datetimes, self._linreg, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"TSI({self.tsi_long},{self.tsi_short})",
                    [self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._tsi_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"OBV EMA({self.obv_ema_period})",
                    [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#66bb6a")],
                ),
            ],
        }
