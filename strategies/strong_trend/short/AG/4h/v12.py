"""StrongTrendShortAG4hV12 — SuperTrend(14,2.8) bearish + MFI(14) < 38.

Economic logic: SuperTrend(14,2.8) on 4H silver uses wider multiplier to filter
noise while maintaining responsiveness. Bearish direction confirms downtrend.
MFI(14) below 38 validates money flowing out — institutional selling pressure
exceeds buying on the 4H timeframe.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.supertrend import supertrend
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV12(TrendingStrategy):
    """SuperTrend(14,2.8) bearish + MFI(14) < 38."""

    name = "strong_trend_short_AG_4h_v12"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    st_period: int = 14
    st_mult: float = 2.8
    mfi_period: int = 14
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes, period=self.st_period, multiplier=self.st_mult,
        )
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes, self.mfi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        sd = self._st_dir[i]
        m = self._mfi[i]

        if np.isnan(sd) or np.isnan(m):
            return 0.0

        if sd >= 0 or m >= 38.0:
            return 0.0

        strength = -0.50
        if m < 25.0:
            strength -= 0.25
        if m < 15.0:
            strength -= 0.15
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": f"SuperTrend({self.st_period},{self.st_mult})", "array": self._st_line, "type": "overlay"},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi,
             "panel": "MFI", "y_range": [0, 100], "horizontal_lines": [20, 38, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st_line, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(f"MFI({self.mfi_period})", [
                    self._make_subplot_trace("MFI", datetimes, self._mfi, color="#42a5f5"),
                ], horizontal_lines=[20, 38, 80], y_range=[0, 100]),
            ],
        }
