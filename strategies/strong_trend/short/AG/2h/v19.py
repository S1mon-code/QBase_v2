"""StrongTrendShortAG2hV19 — Donchian(25) lower break + ROC(14) < 0.

Economic logic: Price breaking the Donchian(25) lower channel on 2H silver
signals a new 25-period low — bearish breakout. ROC(14) negative confirms
that the rate of price change supports the downtrend. Together they filter
false breakdowns from low-volatility consolidation.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.trend.donchian import donchian
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV19(TrendingStrategy):
    """Donchian(25) lower break + ROC(14) < 0."""

    name = "strong_trend_short_AG_2h_v19"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 30

    dc_period: int = 25
    roc_period: int = 14
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_middle, self._dc_lower = donchian(
            self._highs, self._lows, self.dc_period,
        )
        self._roc = rate_of_change(self._closes, self.roc_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        c = self._closes[i]
        dl = self._dc_lower[i]
        r = self._roc[i]

        if any(np.isnan(v) for v in (c, dl, r)):
            return 0.0

        if c > dl or r >= 0:
            return 0.0

        strength = -0.50
        if r < -2.0:
            strength -= 0.25
        if r < -5.0:
            strength -= 0.15
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "DC Upper", "array": self._dc_upper, "type": "overlay"},
            {"name": "DC Middle", "array": self._dc_middle, "type": "overlay"},
            {"name": "DC Lower", "array": self._dc_lower, "type": "overlay"},
            {"name": f"ROC({self.roc_period})", "array": self._roc, "panel": "ROC", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("DC Upper", datetimes, self._dc_upper, color="#ef5350"),
                self._make_overlay("DC Middle", datetimes, self._dc_middle, color="#ffab40"),
                self._make_overlay("DC Lower", datetimes, self._dc_lower, color="#66bb6a"),
            ],
            "subplots": [
                self._make_subplot(f"ROC({self.roc_period})", [
                    self._make_subplot_trace("ROC", datetimes, self._roc, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
