"""StrongTrendLongAG1hV4 — ZLEMA(22) + CCI(22) + CMF(22).

Economic logic: ZLEMA removes lag for fast 1H AG trend detection. CCI with
wider threshold captures Silver's high-volatility momentum. CMF confirms
money flow direction with volume weighting.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.zlema import zlema
from indicators.momentum.cci import cci
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG1hV4(TrendingStrategy):
    name = "strong_trend_long_AG_1h_v4"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    zlema_period: int = 22
    cci_period: int = 22
    cmf_period: int = 22
    cci_threshold: float = 100.0
    chandelier_mult: float = 2.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._zlema = zlema(self._closes, period=self.zlema_period)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        z = self._zlema[bar_index]
        z_prev = self._zlema[bar_index - 1]
        c = self._cci[bar_index]
        m = self._cmf[bar_index]

        if any(np.isnan(v) for v in (z, z_prev, c, m)):
            return 0.0

        if z > z_prev and c > self.cci_threshold and m > 0.0:
            strength = min(1.0, (c - self.cci_threshold) / 100.0 * 0.5 + m * 2.0 + 0.2)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"ZLEMA({self.zlema_period})", "array": self._zlema, "type": "overlay"},
            {"name": f"CCI({self.cci_period})", "array": self._cci,
             "type": "subplot", "zero_line": True,
             "horizontal_lines": [-self.cci_threshold, self.cci_threshold]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf,
             "type": "subplot", "zero_line": True},
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
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#66bb6a")],
                    zero_line=True,
                ),
            ],
        }
