"""StrongTrendShortAG1hV10 — Linear Regression Bearish + TSI Negative + OBV Declining.

Economic logic: Price below linear regression on 1H confirms statistical downtrend.
TSI crossing below zero validates fast momentum shift. OBV declining confirms
cumulative selling volume pressure — ideal for silver's violent 1H moves.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.trend.linear_regression import linear_regression
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV10(TrendingStrategy):
    """LinReg bearish + TSI negative + OBV declining.

    Signal logic:
        close < LinReg AND TSI < -5 AND OBV < OBV_SMA -> -0.85
        close < LinReg AND TSI < 0 -> -0.50
        else -> 0.0
    """

    name = "short_AG_1h_v10"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 40

    lr_period: int = 30
    tsi_long: int = 15
    obv_sma_period: int = 25
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr = linear_regression(self._closes, period=self.lr_period)
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=8,
        )
        self._obv = obv(self._closes, self._volumes)
        n = len(self._obv)
        self._obv_sma = np.full(n, np.nan)
        for i in range(self.obv_sma_period - 1, n):
            w = self._obv[i - self.obv_sma_period + 1 : i + 1]
            v = w[~np.isnan(w)]
            if len(v) > 0:
                self._obv_sma[i] = np.mean(v)

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        lr = self._lr[bar_index]
        t = self._tsi_line[bar_index]
        ov = self._obv[bar_index]
        os_ = self._obv_sma[bar_index]

        if np.isnan(c) or np.isnan(lr) or np.isnan(t):
            return 0.0

        if c >= lr:
            return 0.0

        obv_declining = not np.isnan(ov) and not np.isnan(os_) and ov < os_

        if t < -5 and obv_declining:
            return -0.85
        if t < 0:
            return -0.50
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LinReg({self.lr_period})", "array": self._lr, "type": "overlay"},
            {"name": "TSI", "array": self._tsi_line, "panel": "TSI", "zero_line": True},
            {"name": "TSI Signal", "array": self._tsi_signal, "panel": "TSI"},
            {"name": "OBV", "array": self._obv, "panel": "OBV"},
            {"name": "OBV SMA", "array": self._obv_sma, "panel": "OBV", "style": "dash"},
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
            self._make_subplot("OBV", [
                self._make_subplot_trace("OBV", datetimes, self._obv, color="#26a69a"),
                self._make_subplot_trace("OBV SMA", datetimes, self._obv_sma, style="dash", color="#ab47bc"),
            ]),
        ]
        return {"overlays": overlays, "subplots": subplots}
