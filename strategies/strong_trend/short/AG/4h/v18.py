"""StrongTrendShortAG4hV18 — HMA(38) slope neg + Stochastic(12,5) < 30.

Economic logic: HMA(38) on 4H silver provides low-lag trend detection for the
intermediate timeframe. Negative slope confirms sustained downward movement.
Stochastic(12,5) below 30 validates that price is near its recent lows —
bearish conditions persist without meaningful bounces.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.stochastic import stochastic
from indicators.trend.ema import ema
from indicators.trend.hma import hma
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV18(TrendingStrategy):
    """HMA(38) slope < 0 + Stochastic(12,5) < 30."""

    name = "strong_trend_short_AG_4h_v18"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 45

    hma_period: int = 38
    stoch_k: int = 12
    stoch_d: int = 5
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma_line = hma(self._closes, self.hma_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes, k_period=self.stoch_k, d_period=self.stoch_d,
        )

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        h = self._hma_line[i]
        h_prev = self._hma_line[i - 1] if i > 0 else np.nan
        sk = self._stoch_k[i]

        if any(np.isnan(v) for v in (h, h_prev, sk)):
            return 0.0

        slope = h - h_prev
        if slope >= 0 or sk >= 30.0:
            return 0.0

        strength = -0.50
        if sk < 15.0:
            strength -= 0.25
        if slope < -2.0:
            strength -= 0.15
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma_line, "type": "overlay"},
            {"name": f"Stoch K({self.stoch_k})", "array": self._stoch_k,
             "panel": "Stoch", "y_range": [0, 100], "horizontal_lines": [15, 30, 70]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(f"Stochastic({self.stoch_k},{self.stoch_d})", [
                    self._make_subplot_trace("K", datetimes, self._stoch_k, color="#bb86fc"),
                    self._make_subplot_trace("D", datetimes, self._stoch_d, color="#ffab40"),
                ], horizontal_lines=[15, 30, 70], y_range=[0, 100]),
            ],
        }
