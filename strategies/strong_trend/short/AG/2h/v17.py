"""StrongTrendShortAG2hV17 — Vortex(15) VI- > VI+ + CCI(15) < -50.

Economic logic: Vortex(15) on 2H silver detects directional movement at an
intermediate pace. VI- dominating VI+ confirms bearish momentum. CCI(15)
below -50 validates that price is well below its statistical mean — sustained
selling pressure drives the deviation.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.trend.ema import ema
from indicators.trend.vortex import vortex
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV17(TrendingStrategy):
    """Vortex(15) VI- > VI+ + CCI(15) < -50."""

    name = "strong_trend_short_AG_2h_v17"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 25

    vortex_period: int = 15
    cci_period: int = 15
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._vi_plus, self._vi_minus = vortex(self._highs, self._lows, self._closes, self.vortex_period)
        self._cci = cci(self._highs, self._lows, self._closes, self.cci_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        vp = self._vi_plus[i]
        vm = self._vi_minus[i]
        c = self._cci[i]

        if any(np.isnan(v) for v in (vp, vm, c)):
            return 0.0

        if vm <= vp or c >= -50.0:
            return 0.0

        spread = vm - vp
        strength = -0.40
        if spread > 0.15:
            strength -= 0.25
        if c < -100.0:
            strength -= 0.25
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "VI+", "array": self._vi_plus, "panel": "Vortex", "color": "#66bb6a"},
            {"name": "VI-", "array": self._vi_minus, "panel": "Vortex", "color": "#ef5350"},
            {"name": f"CCI({self.cci_period})", "array": self._cci,
             "panel": "CCI", "horizontal_lines": [-50, -100, 50, 100], "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"Vortex({self.vortex_period})", [
                    self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#66bb6a"),
                    self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                ]),
                self._make_subplot(f"CCI({self.cci_period})", [
                    self._make_subplot_trace("CCI", datetimes, self._cci, color="#42a5f5"),
                ], horizontal_lines=[-50, -100, 50, 100], zero_line=True),
            ],
        }
