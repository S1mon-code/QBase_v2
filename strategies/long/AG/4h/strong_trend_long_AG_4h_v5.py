"""StrongTrendLongAG4hV5 — ZLEMA(35) + CCI(35) + BSP(25).

Economic logic: Zero-Lag EMA removes inherent lag from AG's 4H trend. CCI
with wider threshold detects when Silver trades far from its statistical mean.
Buying/Selling Pressure confirms directional volume dominance.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.zlema import zlema
from indicators.momentum.cci import cci
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG4hV5(TrendingStrategy):
    name = "long_AG_4h_v5"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    zlema_period: int = 35
    cci_period: int = 35
    bsp_period: int = 25
    cci_threshold: float = 100.0
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._zlema = zlema(self._closes, period=self.zlema_period)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._buy_p, self._sell_p = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.bsp_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        z = self._zlema[bar_index]
        z_prev = self._zlema[bar_index - 1]
        c = self._cci[bar_index]
        bp = self._buy_p[bar_index]
        sp = self._sell_p[bar_index]

        if any(np.isnan(v) for v in (z, z_prev, c, bp, sp)):
            return 0.0

        if z > z_prev and c > self.cci_threshold and bp > sp:
            strength = min(1.0, (c - self.cci_threshold) / 150.0 * 0.6 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"ZLEMA({self.zlema_period})", "array": self._zlema, "type": "overlay"},
            {"name": f"CCI({self.cci_period})", "array": self._cci,
             "type": "subplot", "zero_line": True,
             "horizontal_lines": [-self.cci_threshold, self.cci_threshold]},
            {"name": f"BuyPressure({self.bsp_period})", "array": self._buy_p,
             "type": "subplot", "panel": "BSP"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"ZLEMA({self.zlema_period})", datetimes, self._zlema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"CCI({self.cci_period})",
                    [self._make_subplot_trace("CCI", datetimes, self._cci, color="#ab47bc")],
                    zero_line=True, horizontal_lines=[-self.cci_threshold, self.cci_threshold],
                ),
                self._make_subplot(
                    f"BSP({self.bsp_period})",
                    [self._make_subplot_trace("Buy", datetimes, self._buy_p, color="#66bb6a"),
                     self._make_subplot_trace("Sell", datetimes, self._sell_p, color="#ef5350")],
                ),
            ],
        }
