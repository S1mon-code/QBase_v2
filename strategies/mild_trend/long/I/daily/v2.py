"""MildTrendLongIDailyV2 — SuperTrend(80,3.5) + Coppock(120,180,60) + CMF(100).

Economic logic: SuperTrend captures iron ore's structural trend with wide ATR bands
suited to daily volatility. Coppock curve (originally for monthly data) identifies
long-term momentum turns. CMF confirms institutional money flow direction. Signal
strength blends SuperTrend direction, Coppock magnitude, and CMF level.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.coppock import coppock
from indicators.trend.supertrend import supertrend
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV2(TrendingStrategy):
    name = "mild_trend_long_I_daily_v2"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 250

    st_period: int = 80
    st_mult: float = 3.5
    coppock_wma: int = 60
    cmf_period: int = 100
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes, period=self.st_period, multiplier=self.st_mult
        )
        self._coppock = coppock(self._closes, wma_period=self.coppock_wma, roc_long=180, roc_short=120)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        cop = self._coppock[bar_index]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in [st_d, cop, cmf_val]):
            return 0.0

        if st_d != 1.0 or cop <= 0.0 or cmf_val <= 0.0:
            return 0.0

        # Blend: coppock magnitude (capped at 10) + CMF level
        cop_score = min(1.0, cop / 10.0) * 0.5
        cmf_score = min(1.0, max(0.0, cmf_val / 0.3)) * 0.3
        base = 0.2
        return min(1.0, base + cop_score + cmf_score)

    def get_indicator_config(self):
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st_line, "type": "overlay", "style": "step"},
            {"name": f"Coppock({self.coppock_wma})", "array": self._coppock, "type": "subplot", "zero_line": True},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st_line, style="step", color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    "Coppock",
                    [self._make_subplot_trace("Coppock", datetimes, self._coppock, color="#ab47bc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
