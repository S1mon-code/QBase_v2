"""MildTrendLongI2hV2 — SuperTrend(50,2.5) + ROC(40) + OI_Momentum(60).

Economic logic: SuperTrend with moderate parameters tracks 2H iron ore structural
moves. Rate of Change confirms positive price momentum. OI momentum validates
speculative commitment growing with the trend. Signal blends ROC magnitude and
OI expansion for gradual sizing.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.trend.supertrend import supertrend
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV2(TrendingStrategy):
    name = "mild_trend_long_I_2h_v2"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 70

    st_period: int = 50
    st_mult: float = 2.5
    roc_period: int = 40
    oi_period: int = 60
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes, period=self.st_period, multiplier=self.st_mult
        )
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._oi_mom = oi_momentum(self._oi, period=self.oi_period)

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        roc_val = self._roc[bar_index]
        oi_val = self._oi_mom[bar_index]

        if any(np.isnan(v) for v in [st_d, roc_val, oi_val]):
            return 0.0

        if not (st_d == 1.0 and roc_val > 0.0 and oi_val > 0.0):
            return 0.0

        roc_score = min(1.0, roc_val / 5.0) * 0.4
        oi_score = min(1.0, oi_val / 0.10) * 0.3
        return min(1.0, 0.3 + roc_score + oi_score)

    def get_indicator_config(self):
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st_line, "type": "overlay", "style": "step"},
            {"name": f"ROC({self.roc_period})", "array": self._roc, "type": "subplot", "zero_line": True},
            {"name": f"OI Mom({self.oi_period})", "array": self._oi_mom, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st_line, style="step", color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    "ROC",
                    [self._make_subplot_trace("ROC", datetimes, self._roc, color="#42a5f5")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "OI Momentum",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#4fc3f7")],
                    zero_line=True,
                ),
            ],
        }
