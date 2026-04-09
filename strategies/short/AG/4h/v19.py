"""StrongTrendShortAG4hV19 — Donchian(35) lower break + ADX(18) > 20.

Economic logic: Price breaking the Donchian(35) lower channel on 4H silver
signals a new 35-period low — a strong bearish breakout. ADX(18) above 20
confirms the breakout has genuine trend strength, filtering out false
breakdowns in sideways markets.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx
from indicators.trend.donchian import donchian
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV19(TrendingStrategy):
    """Donchian(35) lower break + ADX(18) > 20."""

    name = "short_AG_4h_v19"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 45

    dc_period: int = 35
    adx_period: int = 18
    adx_threshold: float = 20.0
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_middle, self._dc_lower = donchian(
            self._highs, self._lows, self.dc_period,
        )
        self._adx_line = adx(self._highs, self._lows, self._closes, self.adx_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        c = self._closes[i]
        dl = self._dc_lower[i]
        a = self._adx_line[i]

        if any(np.isnan(v) for v in (c, dl, a)):
            return 0.0

        if c > dl or a < self.adx_threshold:
            return 0.0

        adx_str = min(1.0, (a - self.adx_threshold) / 30.0)
        strength = -(0.40 + adx_str * 0.45)
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
            {"name": f"ADX({self.adx_period})", "array": self._adx_line,
             "panel": "ADX", "y_range": [0, 100], "horizontal_lines": [20]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("DC Upper", datetimes, self._dc_upper, color="#ef5350"),
                self._make_overlay("DC Middle", datetimes, self._dc_middle, color="#ffab40"),
                self._make_overlay("DC Lower", datetimes, self._dc_lower, color="#66bb6a"),
            ],
            "subplots": [
                self._make_subplot(f"ADX({self.adx_period})", [
                    self._make_subplot_trace("ADX", datetimes, self._adx_line, color="#42a5f5"),
                ], horizontal_lines=[20], y_range=[0, 100]),
            ],
        }
