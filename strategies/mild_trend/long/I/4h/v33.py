"""MildTrendLongI4hV33 — KAMA(60) slope + Williams %R(16) > -38.

Economic logic: KAMA with medium period adapts to 4h regime shifts. Williams %R
above -38 confirms bullish momentum without extreme overbought conditions. Dual
trend and momentum confirmation for clean 4h entries.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.momentum.williams_r import williams_r
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV33(TrendingStrategy):
    name = "mild_trend_long_I_4h_v33"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup = 80

    kama_period: int = 60
    wr_period: int = 16
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._wr = williams_r(self._highs, self._lows, self._closes, period=self.wr_period)

    def _generate_signal(self, bar_index: int) -> float:
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1] if bar_index > 0 else np.nan
        wr_val = self._wr[bar_index]

        if any(np.isnan(v) for v in [k, k_prev, wr_val]):
            return 0.0

        kama_rising = k > k_prev
        wr_bullish = wr_val > -38.0

        if not (kama_rising and wr_bullish):
            return 0.0

        slope = (k - k_prev) / k_prev * 1000.0 if k_prev != 0 else 0.0
        slope_score = min(0.5, max(0.0, slope / 5.0)) + 0.3
        # Boost with W%R strength
        wr_boost = min(0.2, (wr_val + 38.0) / 190.0)
        return min(1.0, slope_score + wr_boost)

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": f"W%R({self.wr_period})", "array": self._wr, "type": "subplot",
             "y_range": [-100, 0], "horizontal_lines": [-38]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama,
                                   color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Williams %R({self.wr_period})",
                    [self._make_subplot_trace("W%R", datetimes, self._wr, color="#ab47bc")],
                    horizontal_lines=[-38], y_range=[-100, 0],
                ),
            ],
        }
