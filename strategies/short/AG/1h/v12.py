"""StrongTrendShortAG1hV12 — SuperTrend(6,2.0) bearish + CMF(10) < 0.

Economic logic: SuperTrend(6,2.0) on 1H silver uses tight parameters for fast
regime detection. CMF(10) negative confirms distribution — selling volume exceeds
buying volume over the recent window. The fast SuperTrend catches intraday trend
shifts that slower parameters would miss.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.supertrend import supertrend
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV12(TrendingStrategy):
    """SuperTrend(6,2.0) bearish + CMF(10) < 0."""

    name = "short_AG_1h_v12"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 20

    st_period: int = 6
    st_mult: float = 2.0
    cmf_period: int = 10
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes, period=self.st_period, multiplier=self.st_mult,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, i: int) -> float:
        sd = self._st_dir[i]
        cf = self._cmf[i]

        if np.isnan(sd) or np.isnan(cf):
            return 0.0

        if sd >= 0 or cf >= 0:
            return 0.0

        strength = -0.50
        if cf < -0.15:
            strength -= 0.25
        if cf < -0.25:
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
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "panel": "CMF", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st_line, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(f"CMF({self.cmf_period})", [
                    self._make_subplot_trace("CMF", datetimes, self._cmf, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
