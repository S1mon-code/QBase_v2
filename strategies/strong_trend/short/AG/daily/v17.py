"""StrongTrendShortAGDailyV17 — Vortex(22) VI- > VI+ + ROC(20) < 0.

Economic logic: Vortex Indicator VI- exceeding VI+ on daily silver confirms
bearish directional movement dominates. ROC(20) negative validates that price
momentum over 20 days is declining. Together they filter out range-bound
periods where VI may oscillate without true trend.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.trend.ema import ema
from indicators.trend.vortex import vortex
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV17(TrendingStrategy):
    """Vortex(22) VI- > VI+ + ROC(20) < 0."""

    name = "strong_trend_short_AG_daily_v17"
    horizon = "slow"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 30

    vortex_period: int = 22
    roc_period: int = 20
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._vi_plus, self._vi_minus = vortex(self._highs, self._lows, self._closes, self.vortex_period)
        self._roc = rate_of_change(self._closes, self.roc_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, i: int) -> float:
        vp = self._vi_plus[i]
        vm = self._vi_minus[i]
        r = self._roc[i]

        if any(np.isnan(v) for v in (vp, vm, r)):
            return 0.0

        if vm <= vp or r >= 0:
            return 0.0

        spread = vm - vp
        strength = -0.40
        if spread > 0.15:
            strength -= 0.25
        if r < -3.0:
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
            {"name": f"ROC({self.roc_period})", "array": self._roc, "panel": "ROC", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"Vortex({self.vortex_period})", [
                    self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#66bb6a"),
                    self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                ]),
                self._make_subplot(f"ROC({self.roc_period})", [
                    self._make_subplot_trace("ROC", datetimes, self._roc, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
