"""MildTrendShortI1hV20 — TSI bearish + CMF distribution + ADX trending.

Economic logic: TSI(10,20) < 0 on 1H Iron Ore uses double-smoothed momentum
to capture persistent bearish pressure. CMF(10) < 0 confirms distribution.
ADX(10) > 18 ensures directional conviction, filtering out range-bound noise.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.trend.ema import ema
from indicators.trend.adx import adx
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV20(TrendingStrategy):
    """TSI(10,20) < 0 + CMF(10) < 0 + ADX(10) > 18."""

    name = "mild_trend_short_I_1h_v20"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 40

    tsi_fast: int = 10
    tsi_slow: int = 20
    cmf_period: int = 10
    adx_period: int = 10
    chandelier_mult: float = 2.8

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
        self._tsi, _ = tsi(self._closes, long_period=self.tsi_slow, short_period=self.tsi_fast)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)
        self._adx = adx(self._highs, self._lows, self._closes, self.adx_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        t = self._tsi[bar_index]
        c = self._cmf[bar_index]
        a = self._adx[bar_index]

        if any(np.isnan(v) for v in (t, c, a)):
            return 0.0

        if t >= 0:
            return 0.0
        if a <= 18:
            return 0.0

        signal = -(0.3 + min(0.2, abs(t) / 50.0))

        if c < 0:
            signal -= min(0.1, abs(c) * 0.4)

        if a > 25:
            signal -= 0.05

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TSI({self.tsi_fast},{self.tsi_slow})", "array": self._tsi,
             "type": "subplot", "zero_line": True},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
            {"name": f"ADX({self.adx_period})", "array": self._adx, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [18]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"TSI({self.tsi_fast},{self.tsi_slow})",
                    [self._make_subplot_trace("TSI", datetimes, self._tsi, color="#ef5350")],
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
                    horizontal_lines=[18], y_range=[0, 100],
                ),
            ],
        }
