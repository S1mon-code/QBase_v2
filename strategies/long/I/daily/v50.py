"""MildTrendLongIDailyV50 — HMA(80) slope + Stochastic(20,5) > 50 + Force Index(18) > 0.

Economic logic: Hull Moving Average provides a responsive yet smooth trend indication
with minimal lag. Stochastic above 50 confirms bullish momentum mid-range. Positive
Force Index validates that volume and price direction align for buying pressure.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.momentum.stochastic import stochastic
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV50(TrendingStrategy):
    name = "long_I_daily_v50"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 120

    hma_period: int = 80
    stoch_k: int = 20
    stoch_d: int = 5
    fi_period: int = 18
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes,
            k_period=self.stoch_k, d_period=self.stoch_d)
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1] if bar_index > 0 else np.nan
        sk = self._stoch_k[bar_index]
        fi_val = self._fi[bar_index]

        if any(np.isnan(v) for v in [h, h_prev, sk, fi_val]):
            return 0.0

        hma_rising = h > h_prev
        stoch_bullish = sk > 50.0
        fi_positive = fi_val > 0.0

        if not (hma_rising and stoch_bullish and fi_positive):
            return 0.0

        # Scale by HMA slope magnitude
        slope = (h - h_prev) / h_prev * 1000.0 if h_prev != 0 else 0.0
        slope_score = min(0.5, max(0.0, slope / 5.0)) + 0.3
        # Boost with stochastic
        stoch_boost = min(0.2, (sk - 50.0) / 250.0)
        return min(1.0, slope_score + stoch_boost)

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": f"Stoch K({self.stoch_k})", "array": self._stoch_k, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [50]},
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "type": "subplot",
             "horizontal_lines": [0]},
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
                    horizontal_lines=[50], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"Force Index({self.fi_period})",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#66bb6a")],
                    horizontal_lines=[0],
                ),
            ],
        }
