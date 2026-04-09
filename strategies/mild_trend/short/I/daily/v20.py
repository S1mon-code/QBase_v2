"""MildTrendShortIDailyV20 — TRIX bearish + CMF distribution + ADX trending.

Economic logic: TRIX(20) < 0 on daily Iron Ore is a triple-smoothed momentum
indicator signaling persistent bearish pressure. CMF(30) < 0 confirms
distribution. ADX(20) > 20 ensures the move has directional conviction,
filtering out choppy regimes.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.trix import trix
from indicators.trend.ema import ema
from indicators.trend.adx import adx
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV20(TrendingStrategy):
    """TRIX(20) < 0 + CMF(30) < 0 + ADX(20) > 20."""

    name = "mild_trend_short_I_daily_v20"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    trix_period: int = 20
    cmf_period: int = 30
    adx_period: int = 20
    adx_threshold: float = 20.0
    chandelier_mult: float = 3.5

    def on_init_arrays(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        volumes: np.ndarray,
        oi: np.ndarray,
        datetimes: np.ndarray,
    ) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._trix, self._trix_signal = trix(self._closes, self.trix_period)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)
        self._adx = adx(self._highs, self._lows, self._closes, self.adx_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        t = self._trix[bar_index]
        c = self._cmf[bar_index]
        a = self._adx[bar_index]

        if any(np.isnan(v) for v in (t, c, a)):
            return 0.0

        if t >= 0:
            return 0.0
        if a <= self.adx_threshold:
            return 0.0

        # Base from TRIX magnitude
        signal = -(0.3 + min(0.2, abs(t) * 50))

        # CMF distribution
        if c < 0:
            signal -= min(0.1, abs(c) * 0.4)

        # ADX strength bonus
        if a > 30:
            signal -= 0.1

        return max(-0.7, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TRIX({self.trix_period})", "array": self._trix, "type": "subplot", "zero_line": True},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
            {"name": f"ADX({self.adx_period})", "array": self._adx, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [self.adx_threshold]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"TRIX({self.trix_period})",
                    [self._make_subplot_trace("TRIX", datetimes, self._trix, color="#ef5350")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26c6da")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"ADX({self.adx_period})",
                    [self._make_subplot_trace("ADX", datetimes, self._adx, color="#bb86fc")],
                    horizontal_lines=[self.adx_threshold], y_range=[0, 100],
                ),
            ],
        }
