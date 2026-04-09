"""MildTrendLongI1hV3 — SuperTrend(25,2.2) + Fisher(20) + OI_Momentum(30).

Economic logic: SuperTrend with tight parameters captures 1H iron ore intraday trends.
Fisher Transform normalizes price — positive values confirm bullish state.
OI momentum validates speculative interest growth. Signal scales with Fisher magnitude
and OI expansion rate.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.fisher_transform import fisher_transform
from indicators.trend.supertrend import supertrend
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV3(TrendingStrategy):
    name = "mild_trend_long_I_1h_v3"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 40

    st_period: int = 25
    st_mult: float = 2.2
    fisher_period: int = 20
    oi_period: int = 30
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes, period=self.st_period, multiplier=self.st_mult
        )
        self._fisher, self._fisher_trigger = fisher_transform(
            self._highs, self._lows, period=self.fisher_period
        )
        self._oi_mom = oi_momentum(self._oi, period=self.oi_period)

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        f_val = self._fisher[bar_index]
        oi_val = self._oi_mom[bar_index]

        if any(np.isnan(v) for v in [st_d, f_val, oi_val]):
            return 0.0

        if not (st_d == 1.0 and f_val > 0.0 and oi_val > 0.0):
            return 0.0

        fisher_score = min(1.0, f_val / 2.0) * 0.4
        oi_score = min(1.0, oi_val / 0.08) * 0.3
        return min(1.0, 0.3 + fisher_score + oi_score)

    def get_indicator_config(self):
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st_line, "type": "overlay", "style": "step"},
            {"name": "Fisher", "array": self._fisher, "type": "subplot", "panel": "Fisher", "zero_line": True},
            {"name": "Trigger", "array": self._fisher_trigger, "type": "subplot", "panel": "Fisher", "style": "dash"},
            {"name": f"OI Mom({self.oi_period})", "array": self._oi_mom, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st_line, style="step", color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    "Fisher Transform",
                    [
                        self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#bb86fc"),
                        self._make_subplot_trace("Trigger", datetimes, self._fisher_trigger, color="#ff8a80", style="dash"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "OI Momentum",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#4fc3f7")],
                    zero_line=True,
                ),
            ],
        }
