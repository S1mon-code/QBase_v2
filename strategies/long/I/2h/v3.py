"""MildTrendLongI2hV3 — KAMA(60) + Williams%R(40) + CMF(60).

Economic logic: KAMA adapts to 2H iron ore regime shifts. Williams %R above -50
confirms price in upper half of recent range (bullish). CMF validates volume-weighted
buying pressure. Signal scales with Williams %R distance from midpoint and CMF level.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.williams_r import williams_r
from indicators.trend.kama import kama
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV3(TrendingStrategy):
    name = "long_I_2h_v3"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 70

    kama_period: int = 60
    wr_period: int = 40
    cmf_period: int = 60
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._wr = williams_r(self._highs, self._lows, self._closes, period=self.wr_period)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1] if bar_index > 0 else np.nan
        wr = self._wr[bar_index]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in [k, k_prev, wr, cmf_val]):
            return 0.0

        kama_rising = k > k_prev
        wr_bullish = wr > -50.0  # Upper half of range
        cmf_positive = cmf_val > 0.0

        if not (kama_rising and wr_bullish and cmf_positive):
            return 0.0

        # Williams %R: -50 to 0 maps to 0.3 to 0.7
        wr_score = min(1.0, (wr + 50.0) / 50.0) * 0.4
        cmf_score = min(1.0, cmf_val / 0.25) * 0.3
        return min(1.0, 0.3 + wr_score + cmf_score)

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": f"Williams %R({self.wr_period})", "array": self._wr, "type": "subplot",
             "y_range": [-100, 0], "horizontal_lines": [-20, -80]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Williams %R",
                    [self._make_subplot_trace("%R", datetimes, self._wr, color="#bb86fc")],
                    y_range=[-100, 0], horizontal_lines=[-20, -50, -80],
                ),
                self._make_subplot(
                    "CMF",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
