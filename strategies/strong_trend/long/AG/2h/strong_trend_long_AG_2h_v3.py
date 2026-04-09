"""StrongTrendLongAG2hV3 — HMA(45) + Schaff(50,30,12) + ForceIndex(35).

Economic logic: HMA provides minimal-lag trend on AG 2H. Schaff Trend Cycle
applies double stochastic smoothing to detect clean cycle turns. Force Index
confirms volume-backed directional movement.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG2hV3(TrendingStrategy):
    name = "strong_trend_long_AG_2h_v3"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    hma_period: int = 45
    schaff_period: int = 50
    schaff_fast: int = 30
    schaff_slow: int = 12
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._schaff = schaff_trend_cycle(
            self._closes, period=self.schaff_period,
            fast=self.schaff_fast, slow=self.schaff_slow,
        )
        self._force = force_index(self._closes, self._volumes, period=35)

    def _generate_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1]
        sc = self._schaff[bar_index]
        fi = self._force[bar_index]

        if any(np.isnan(v) for v in (h, h_prev, sc, fi)):
            return 0.0

        if h > h_prev and sc > 50.0 and fi > 0.0:
            strength = min(1.0, (sc - 50.0) / 50.0 * 0.6 + 0.3)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": f"Schaff({self.schaff_period})", "array": self._schaff,
             "type": "subplot", "y_range": [0, 100], "horizontal_lines": [25, 50, 75]},
            {"name": "ForceIndex(35)", "array": self._force,
             "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Schaff({self.schaff_period})",
                    [self._make_subplot_trace("STC", datetimes, self._schaff, color="#ab47bc")],
                    y_range=[0, 100], horizontal_lines=[25, 50, 75],
                ),
                self._make_subplot(
                    "ForceIndex(35)",
                    [self._make_subplot_trace("Force", datetimes, self._force, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
