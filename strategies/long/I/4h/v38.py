"""MildTrendLongI4hV38 — ROC(20) > 0 + SuperTrend(12,3.0) bullish + CMF(18) > 0.

Economic logic: ROC captures momentum over medium window on 4h bars. SuperTrend
provides adaptive trend confirmation. CMF validates institutional money flow. Triple
confirmation for high-conviction 4h trend entries.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.trend.supertrend import supertrend
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV38(TrendingStrategy):
    name = "long_I_4h_v38"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 40

    roc_period: int = 20
    st_period: int = 12
    st_mult: float = 3.0
    cmf_period: int = 18
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._roc = rate_of_change(self._closes, self.roc_period)
        self._st, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes,
                        period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        roc_val = self._roc[bar_index]
        st_dir = self._st_dir[bar_index]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in [roc_val, st_dir, cmf_val]):
            return 0.0

        roc_positive = roc_val > 0.0
        st_bullish = st_dir > 0.0
        cmf_positive = cmf_val > 0.0

        if not (roc_positive and st_bullish and cmf_positive):
            return 0.0

        roc_score = min(0.5, max(0.0, roc_val / 20.0)) + 0.3
        cmf_boost = min(0.2, max(0.0, cmf_val))
        return min(1.0, roc_score + cmf_boost)

    def get_indicator_config(self):
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st, "type": "overlay"},
            {"name": f"ROC({self.roc_period})", "array": self._roc, "type": "subplot",
             "horizontal_lines": [0]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot",
             "horizontal_lines": [0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st,
                                   color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    f"ROC({self.roc_period})",
                    [self._make_subplot_trace("ROC", datetimes, self._roc, color="#42a5f5")],
                    horizontal_lines=[0],
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26a69a")],
                    horizontal_lines=[0],
                ),
            ],
        }
