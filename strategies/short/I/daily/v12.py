"""MildTrendShortIDailyV12 — SuperTrend bearish + CMF distribution.

Economic logic: SuperTrend(20,3.5) flipping bearish on daily Iron Ore signals
a regime shift from consolidation to downtrend. CMF(25) < 0 confirms
institutional money flowing out, validating the bearish SuperTrend signal
with volume-based distribution evidence.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.trend.ema import ema
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV12(TrendingStrategy):
    """SuperTrend(20,3.5) bearish + CMF(25) < 0."""

    name = "short_I_daily_v12"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    st_period: int = 20
    st_mult: float = 3.5
    cmf_period: int = 25
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
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes, self.st_period, self.st_mult,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        c = self._cmf[bar_index]
        price = self._closes[bar_index]
        st_val = self._st_line[bar_index]

        if any(np.isnan(v) for v in (st_d, c, st_val)):
            return 0.0

        # SuperTrend must be bearish (direction == -1)
        if st_d >= 0:
            return 0.0

        # Distance from SuperTrend line
        dist = (st_val - price) / price if price != 0 else 0.0
        signal = -(0.3 + min(0.2, dist * 3.0))

        # CMF distribution confirmation
        if c < 0:
            signal -= min(0.15, abs(c) * 0.5)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"SuperTrend({self.st_period},{self.st_mult})", "array": self._st_line,
             "type": "overlay", "color": "#ef5350"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period},{self.st_mult})", datetimes, self._st_line, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26c6da")],
                    zero_line=True,
                ),
            ],
        }
