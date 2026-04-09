"""StrongTrendShortAG4hV5 — Linear Regression Bearish + TSI Negative + Force Index.

Economic logic: Price below linear regression line confirms the statistical
downtrend. TSI crossing below zero validates momentum shift. Negative Force
Index confirms selling volume dominates.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.trend.linear_regression import linear_regression
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV5(TrendingStrategy):
    """LinReg bearish + TSI negative + Force Index sell pressure.

    Signal logic:
        close < LinReg AND TSI < 0 AND ForceIndex < 0 -> -0.85
        close < LinReg AND TSI < 0 -> -0.45
        else -> 0.0
    """

    name = "short_AG_4h_v5"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    lr_period: int = 60
    tsi_long: int = 35
    fi_period: int = 25
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr = linear_regression(self._closes, period=self.lr_period)
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=18,
        )
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        lr = self._lr[bar_index]
        t = self._tsi_line[bar_index]
        fi = self._fi[bar_index]

        if np.isnan(c) or np.isnan(lr) or np.isnan(t):
            return 0.0

        if c >= lr:
            return 0.0

        if t < 0 and (not np.isnan(fi) and fi < 0):
            return -0.85
        if t < 0:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LinReg({self.lr_period})", "array": self._lr, "type": "overlay"},
            {"name": "TSI", "array": self._tsi_line, "panel": "TSI", "zero_line": True},
            {"name": "TSI Signal", "array": self._tsi_signal, "panel": "TSI"},
            {"name": f"ForceIndex({self.fi_period})", "array": self._fi, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"LinReg({self.lr_period})", datetimes, self._lr, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot("TSI", [
                self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#42a5f5"),
                self._make_subplot_trace("TSI Signal", datetimes, self._tsi_signal, color="#ff7043"),
            ], zero_line=True),
            self._make_subplot(f"ForceIndex({self.fi_period})", [
                self._make_subplot_trace("ForceIndex", datetimes, self._fi, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
