"""MildTrendLongI1hV60 — Aroon(10) up > 70 + TRIX(8) bullish + CMF(8) > 0.

Economic logic: Aroon up above 70 means price made a new high recently — trend is
active. TRIX triple-smoothed momentum filters noise on 1h bars. Positive CMF confirms
money flow supports the move. Triple confirmation for high-conviction intraday signals.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.momentum.trix import trix
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV60(TrendingStrategy):
    name = "long_I_1h_v60"
    horizon = "fast"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 30

    aroon_period: int = 10
    trix_period: int = 8
    cmf_period: int = 8
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._aroon_up, self._aroon_down, _ = aroon(self._highs, self._lows,
                                                     self.aroon_period)
        self._trix, self._trix_signal = trix(self._closes, period=self.trix_period)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes,
                        period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        aroon_up = self._aroon_up[bar_index]
        trix_val = self._trix[bar_index]
        trix_prev = self._trix[bar_index - 1] if bar_index > 0 else np.nan
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in [aroon_up, trix_val, trix_prev, cmf_val]):
            return 0.0

        aroon_strong = aroon_up > 70.0
        trix_bullish = trix_val > trix_prev
        cmf_positive = cmf_val > 0.0

        if not (aroon_strong and trix_bullish and cmf_positive):
            return 0.0

        # Scale by Aroon strength (70-100 mapped to 0.3-0.8)
        aroon_score = min(0.5, max(0.0, (aroon_up - 70.0) / 60.0)) + 0.3
        cmf_boost = min(0.2, max(0.0, cmf_val))
        return min(1.0, aroon_score + cmf_boost)

    def get_indicator_config(self):
        return [
            {"name": f"Aroon Up({self.aroon_period})", "array": self._aroon_up,
             "type": "subplot", "y_range": [0, 100], "horizontal_lines": [70]},
            {"name": f"TRIX({self.trix_period})", "array": self._trix, "type": "subplot",
             "horizontal_lines": [0]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot",
             "horizontal_lines": [0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"Aroon({self.aroon_period})",
                    [
                        self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#26a69a"),
                        self._make_subplot_trace("Down", datetimes, self._aroon_down,
                                                 color="#ef5350"),
                    ],
                    horizontal_lines=[70], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"TRIX({self.trix_period})",
                    [
                        self._make_subplot_trace("TRIX", datetimes, self._trix, color="#ab47bc"),
                        self._make_subplot_trace("Signal", datetimes, self._trix_signal,
                                                 color="#ff7043", style="dash"),
                    ],
                    horizontal_lines=[0],
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#42a5f5")],
                    horizontal_lines=[0],
                ),
            ],
        }
