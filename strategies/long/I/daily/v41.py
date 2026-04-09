"""MildTrendLongIDailyV41 — Schaff Trend Cycle(30,20,10) + CMF(25).

Economic logic: Schaff Trend Cycle combines MACD with stochastic smoothing to
detect momentum shifts earlier than raw MACD. When STC rises above 25, a new
uptrend is forming. CMF confirms that institutional money flow supports the move.
Signal scales with STC strength and CMF magnitude.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV41(TrendingStrategy):
    name = "long_I_daily_v41"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 80

    # Optimizable parameters (<=5 including chandelier_mult)
    stc_period: int = 30
    stc_fast: int = 20
    stc_slow: int = 10
    cmf_period: int = 25
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._stc = schaff_trend_cycle(self._closes, self.stc_period,
                                       fast=self.stc_fast, slow=self.stc_slow)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes,
                        period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        stc_val = self._stc[bar_index]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in [stc_val, cmf_val]):
            return 0.0

        stc_bullish = stc_val > 25.0
        cmf_positive = cmf_val > 0.0

        if not (stc_bullish and cmf_positive):
            return 0.0

        # Scale by STC strength (25-75 range mapped to 0.3-1.0)
        stc_score = min(1.0, max(0.0, (stc_val - 25.0) / 50.0)) * 0.5 + 0.3
        # Boost with CMF magnitude
        cmf_score = min(0.2, max(0.0, cmf_val))
        return min(1.0, stc_score + cmf_score)

    def get_indicator_config(self):
        return [
            {"name": f"STC({self.stc_period})", "array": self._stc, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [25, 75]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot",
             "y_range": [-0.5, 0.5], "horizontal_lines": [0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"STC({self.stc_period})",
                    [self._make_subplot_trace("STC", datetimes, self._stc, color="#ab47bc")],
                    horizontal_lines=[25, 75], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26a69a")],
                    horizontal_lines=[0], y_range=[-0.5, 0.5],
                ),
            ],
        }
