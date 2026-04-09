"""MildTrendLongI2hV40 — Aroon(16) up > 70 + TRIX(12) bullish.

Economic logic: Aroon up above 70 on 2h bars indicates recent new highs — trend is
active. TRIX triple-smoothed momentum confirms direction and filters noise. Dual
trend and momentum confirmation for robust entries.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.momentum.trix import trix
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV40(TrendingStrategy):
    name = "mild_trend_long_I_2h_v40"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup = 35

    aroon_period: int = 16
    trix_period: int = 12
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._aroon_up, self._aroon_down, _ = aroon(self._highs, self._lows,
                                                     self.aroon_period)
        self._trix, self._trix_signal = trix(self._closes, period=self.trix_period)

    def _generate_signal(self, bar_index: int) -> float:
        aroon_up = self._aroon_up[bar_index]
        trix_val = self._trix[bar_index]
        trix_prev = self._trix[bar_index - 1] if bar_index > 0 else np.nan

        if any(np.isnan(v) for v in [aroon_up, trix_val, trix_prev]):
            return 0.0

        aroon_strong = aroon_up > 70.0
        trix_bullish = trix_val > trix_prev

        if not (aroon_strong and trix_bullish):
            return 0.0

        aroon_score = min(0.5, max(0.0, (aroon_up - 70.0) / 60.0)) + 0.3
        # Boost if TRIX is also positive
        if trix_val > 0.0:
            aroon_score += 0.15
        return min(1.0, aroon_score)

    def get_indicator_config(self):
        return [
            {"name": f"Aroon Up({self.aroon_period})", "array": self._aroon_up,
             "type": "subplot", "y_range": [0, 100], "horizontal_lines": [70]},
            {"name": f"TRIX({self.trix_period})", "array": self._trix, "type": "subplot",
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
            ],
        }
