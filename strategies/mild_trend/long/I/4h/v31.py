"""MildTrendLongI4hV31 — Schaff Trend(25,16,8) > 25 + CMF(20) > 0.

Economic logic: Schaff Trend Cycle on 4h with medium parameters balances signal
quality and responsiveness. STC above 25 signals emerging uptrend. CMF confirms
institutional money flow direction. Signal scales with STC strength and CMF.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV31(TrendingStrategy):
    name = "mild_trend_long_I_4h_v31"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 60

    stc_period: int = 25
    stc_fast: int = 16
    stc_slow: int = 8
    cmf_period: int = 20
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

        if not (stc_val > 25.0 and cmf_val > 0.0):
            return 0.0

        stc_score = min(1.0, max(0.0, (stc_val - 25.0) / 50.0)) * 0.5 + 0.3
        cmf_score = min(0.2, max(0.0, cmf_val))
        return min(1.0, stc_score + cmf_score)

    def get_indicator_config(self):
        return [
            {"name": f"STC({self.stc_period})", "array": self._stc, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [25, 75]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot",
             "horizontal_lines": [0]},
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
                    horizontal_lines=[0],
                ),
            ],
        }
