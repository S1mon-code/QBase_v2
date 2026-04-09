"""MildTrendShortI4hV2 — SuperTrend Bearish + RSI Weak + CMF Distribution.

Economic logic: SuperTrend bearish state on 4H confirms sustained downtrend.
RSI below 50 validates momentum bias. CMF below zero shows money flowing
out of the asset, confirming distribution.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.momentum.rsi import rsi
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV2(TrendingStrategy):
    """SuperTrend(35,2.8) bearish + RSI(30)<50 + CMF(40)<0."""

    name = "mild_trend_short_I_4h_v2"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 55

    st_period: int = 35
    st_mult: float = 2.8
    rsi_period: int = 30
    cmf_period: int = 40
    chandelier_mult: float = 2.5

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
        self._rsi = rsi(self._closes, self.rsi_period)
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes, self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        rsi_val = self._rsi[bar_index]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in (st_d, rsi_val)):
            return 0.0

        if st_d > 0:
            return 0.0

        signal = -0.3

        if rsi_val < 50:
            rsi_str = min(1.0, (50.0 - rsi_val) / 30.0)
            signal -= 0.25 * rsi_str

        if not np.isnan(cmf_val) and cmf_val < 0:
            signal -= 0.2

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st_line, "type": "overlay", "color": "#ef5350"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 50, 70]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period},{self.st_mult})", datetimes, self._st_line, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"RSI({self.rsi_period})",
                    [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc")],
                    horizontal_lines=[30, 50, 70], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
