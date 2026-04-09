"""StrongTrendLongAG2hV7 — ZLEMA(40) + CCI(40) + OBV(50).

Economic logic: Zero-Lag EMA removes delay on AG 2H trend tracking. CCI
with wider threshold captures Silver's high-volatility deviations from mean.
OBV confirms cumulative volume pressure direction.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.zlema import zlema
from indicators.momentum.cci import cci
from indicators.volume.obv import obv
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG2hV7(TrendingStrategy):
    name = "long_AG_2h_v7"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    zlema_period: int = 40
    cci_period: int = 40
    obv_ema_period: int = 50
    cci_threshold: float = 100.0
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._zlema = zlema(self._closes, period=self.zlema_period)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        raw_obv = obv(self._closes, self._volumes)
        self._obv_ema = ema(raw_obv, period=self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        z = self._zlema[bar_index]
        z_prev = self._zlema[bar_index - 1]
        c = self._cci[bar_index]
        oe = self._obv_ema[bar_index]
        oe_prev = self._obv_ema[bar_index - 1]

        if any(np.isnan(v) for v in (z, z_prev, c, oe, oe_prev)):
            return 0.0

        if z > z_prev and c > self.cci_threshold and oe > oe_prev:
            strength = min(1.0, (c - self.cci_threshold) / 150.0 * 0.6 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"ZLEMA({self.zlema_period})", "array": self._zlema, "type": "overlay"},
            {"name": f"CCI({self.cci_period})", "array": self._cci,
             "type": "subplot", "zero_line": True,
             "horizontal_lines": [-self.cci_threshold, self.cci_threshold]},
            {"name": f"OBV EMA({self.obv_ema_period})", "array": self._obv_ema, "type": "subplot"},
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
                    f"OBV EMA({self.obv_ema_period})",
                    [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#66bb6a")],
                ),
            ],
        }
