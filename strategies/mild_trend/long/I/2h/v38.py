"""MildTrendLongI2hV38 — ROC(16) > 0 + SuperTrend(10,2.8) bullish.

Economic logic: ROC captures momentum over medium window on 2h bars. SuperTrend
with balanced parameters provides adaptive trend confirmation. Dual agreement for
clean trend-following entries.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.trend.supertrend import supertrend
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV38(TrendingStrategy):
    name = "mild_trend_long_I_2h_v38"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup = 30

    roc_period: int = 16
    st_period: int = 10
    st_mult: float = 2.8
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._roc = rate_of_change(self._closes, self.roc_period)
        self._st, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult)

    def _generate_signal(self, bar_index: int) -> float:
        roc_val = self._roc[bar_index]
        st_dir = self._st_dir[bar_index]

        if any(np.isnan(v) for v in [roc_val, st_dir]):
            return 0.0

        roc_positive = roc_val > 0.0
        st_bullish = st_dir > 0.0

        if not (roc_positive and st_bullish):
            return 0.0

        roc_score = min(0.7, max(0.0, roc_val / 10.0)) + 0.3
        return min(1.0, roc_score)

    def get_indicator_config(self):
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st, "type": "overlay"},
            {"name": f"ROC({self.roc_period})", "array": self._roc, "type": "subplot",
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
            ],
        }
