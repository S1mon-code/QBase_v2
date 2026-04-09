"""StrongTrendShortAG4hV15 — KAMA(55) slope neg + CCI(18) < -50.

Economic logic: KAMA(55) on 4H silver adapts to volatility while tracking
the longer-term trend. Negative slope confirms sustained bearish direction.
CCI(18) below -50 validates that price deviates significantly below its
statistical mean — persistent selling drives the downturn.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.trend.ema import ema
from indicators.trend.kama import kama
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV15(TrendingStrategy):
    """KAMA(55) slope < 0 + CCI(18) < -50."""

    name = "strong_trend_short_AG_4h_v15"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 65

    kama_period: int = 55
    cci_period: int = 18
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama_line = kama(self._closes, self.kama_period)
        self._cci = cci(self._highs, self._lows, self._closes, self.cci_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        k = self._kama_line[i]
        k_prev = self._kama_line[i - 1] if i > 0 else np.nan
        c = self._cci[i]

        if any(np.isnan(v) for v in (k, k_prev, c)):
            return 0.0

        slope = k - k_prev
        if slope >= 0 or c >= -50.0:
            return 0.0

        strength = -0.45
        if c < -100.0:
            strength -= 0.25
        if slope < -1.0:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama_line, "type": "overlay"},
            {"name": f"CCI({self.cci_period})", "array": self._cci,
             "panel": "CCI", "horizontal_lines": [-50, -100, 50, 100], "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(f"CCI({self.cci_period})", [
                    self._make_subplot_trace("CCI", datetimes, self._cci, color="#42a5f5"),
                ], horizontal_lines=[-50, -100, 50, 100], zero_line=True),
            ],
        }
