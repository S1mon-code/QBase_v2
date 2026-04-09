"""MildTrendLongI4hV39 — HMA(50) slope + Stochastic(14,5) > 52.

Economic logic: Hull Moving Average on 4h provides responsive trend detection with
minimal lag at medium-term horizons. Stochastic above 52 confirms moderate bullish
momentum. Signal scales with HMA slope magnitude and stochastic level.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.momentum.stochastic import stochastic
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV39(TrendingStrategy):
    name = "long_I_4h_v39"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup = 75

    hma_period: int = 50
    stoch_k: int = 14
    stoch_d: int = 5
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes,
            k_period=self.stoch_k, d_period=self.stoch_d)

    def _generate_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1] if bar_index > 0 else np.nan
        sk = self._stoch_k[bar_index]

        if any(np.isnan(v) for v in [h, h_prev, sk]):
            return 0.0

        hma_rising = h > h_prev
        stoch_bullish = sk > 52.0

        if not (hma_rising and stoch_bullish):
            return 0.0

        slope = (h - h_prev) / h_prev * 1000.0 if h_prev != 0 else 0.0
        slope_score = min(0.5, max(0.0, slope / 5.0)) + 0.3
        stoch_boost = min(0.2, (sk - 52.0) / 240.0)
        return min(1.0, slope_score + stoch_boost)

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": f"Stoch K({self.stoch_k})", "array": self._stoch_k, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [52]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma,
                                   color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Stochastic({self.stoch_k},{self.stoch_d})",
                    [
                        self._make_subplot_trace("%K", datetimes, self._stoch_k, color="#42a5f5"),
                        self._make_subplot_trace("%D", datetimes, self._stoch_d,
                                                 color="#ef5350", style="dash"),
                    ],
                    horizontal_lines=[52], y_range=[0, 100],
                ),
            ],
        }
