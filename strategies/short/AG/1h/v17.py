"""StrongTrendShortAG1hV17 — Vortex(10) VI- > VI+ + Williams %R(8) < -60.

Economic logic: Vortex(10) on 1H silver detects directional movement shifts
quickly. VI- dominating VI+ confirms bearish momentum. Williams %R(8) below -60
validates that price is trading in the lower portion of its recent range,
reinforcing the short bias.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.williams_r import williams_r
from indicators.trend.ema import ema
from indicators.trend.vortex import vortex
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV17(TrendingStrategy):
    """Vortex(10) VI- > VI+ + Williams %R(8) < -60."""

    name = "short_AG_1h_v17"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 15

    vortex_period: int = 10
    wr_period: int = 8
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._vi_plus, self._vi_minus = vortex(self._highs, self._lows, self._closes, self.vortex_period)
        self._wr = williams_r(self._highs, self._lows, self._closes, self.wr_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, i: int) -> float:
        vp = self._vi_plus[i]
        vm = self._vi_minus[i]
        wr = self._wr[i]

        if any(np.isnan(v) for v in (vp, vm, wr)):
            return 0.0

        if vm <= vp or wr >= -60.0:
            return 0.0

        spread = vm - vp
        strength = -0.45
        if spread > 0.15:
            strength -= 0.25
        if wr < -80.0:
            strength -= 0.20
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
            {"name": f"Williams %R({self.wr_period})", "array": self._wr,
             "panel": "WR", "y_range": [-100, 0], "horizontal_lines": [-60, -80]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"Vortex({self.vortex_period})", [
                    self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#66bb6a"),
                    self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                ]),
                self._make_subplot(f"Williams %R({self.wr_period})", [
                    self._make_subplot_trace("WR", datetimes, self._wr, color="#bb86fc"),
                ], horizontal_lines=[-60, -80], y_range=[-100, 0]),
            ],
        }
