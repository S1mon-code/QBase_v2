"""StrongTrendShortAGDailyV18 — HMA(50) slope neg + CCI(22) < -50.

Economic logic: HMA(50) negative slope detects trend changes faster than SMA/EMA
due to its reduced lag. CCI(22) below -50 confirms silver is trading well below
its statistical mean — bearish momentum is sustained. The lag-free HMA avoids
late entries common in traditional moving average strategies.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.trend.ema import ema
from indicators.trend.hma import hma
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV18(TrendingStrategy):
    """HMA(50) slope < 0 + CCI(22) < -50."""

    name = "short_AG_daily_v18"
    horizon = "slow"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 60

    hma_period: int = 50
    cci_period: int = 22
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma_line = hma(self._closes, self.hma_period)
        self._cci = cci(self._highs, self._lows, self._closes, self.cci_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, i: int) -> float:
        h = self._hma_line[i]
        h_prev = self._hma_line[i - 1] if i > 0 else np.nan
        c = self._cci[i]

        if any(np.isnan(v) for v in (h, h_prev, c)):
            return 0.0

        slope = h - h_prev
        if slope >= 0 or c >= -50.0:
            return 0.0

        strength = -0.45
        if c < -100.0:
            strength -= 0.25
        if slope < -2.0:
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
            {"name": f"CCI({self.cci_period})", "array": self._cci,
             "panel": "CCI", "horizontal_lines": [-50, -100, 50, 100], "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(f"CCI({self.cci_period})", [
                    self._make_subplot_trace("CCI", datetimes, self._cci, color="#42a5f5"),
                ], horizontal_lines=[-50, -100, 50, 100], zero_line=True),
            ],
        }
