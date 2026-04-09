"""StrongTrendShortAG2hV18 — HMA(28) slope neg + Williams %R(12) < -55.

Economic logic: HMA(28) on 2H silver provides low-lag trend detection at an
intermediate timeframe. Negative slope confirms downward direction. Williams
%R(12) below -55 validates that price is in the lower portion of its recent
range — sellers maintain control.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.williams_r import williams_r
from indicators.trend.ema import ema
from indicators.trend.hma import hma
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV18(TrendingStrategy):
    """HMA(28) slope < 0 + Williams %R(12) < -55."""

    name = "strong_trend_short_AG_2h_v18"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 35

    hma_period: int = 28
    wr_period: int = 12
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma_line = hma(self._closes, self.hma_period)
        self._wr = williams_r(self._highs, self._lows, self._closes, self.wr_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        h = self._hma_line[i]
        h_prev = self._hma_line[i - 1] if i > 0 else np.nan
        wr = self._wr[i]

        if any(np.isnan(v) for v in (h, h_prev, wr)):
            return 0.0

        slope = h - h_prev
        if slope >= 0 or wr >= -55.0:
            return 0.0

        strength = -0.45
        if wr < -75.0:
            strength -= 0.25
        if slope < -1.5:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma_line, "type": "overlay"},
            {"name": f"Williams %R({self.wr_period})", "array": self._wr,
             "panel": "WR", "y_range": [-100, 0], "horizontal_lines": [-55, -80]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(f"Williams %R({self.wr_period})", [
                    self._make_subplot_trace("WR", datetimes, self._wr, color="#bb86fc"),
                ], horizontal_lines=[-55, -80], y_range=[-100, 0]),
            ],
        }
