"""StrongTrendShortAG1hV2 — SuperTrend(15,3) bearish + MACD(8,21,5) + OBV declining.

Economic logic: SuperTrend direction provides a hysteresis-based trend filter
that inherently resists whipsaw (only flips when price crosses the ATR band).
MACD *line* (not histogram) below zero confirms medium-term momentum is bearish.
Declining OBV validates that volume is leaving during selloffs.  Signal smoothed
with 3-bar EMA.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.ema import ema
from indicators.trend.supertrend import supertrend
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV2(TrendingStrategy):
    """SuperTrend bearish + MACD line < 0 + OBV declining."""

    name = "short_AG_1h_v2"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    st_period: int = 15
    st_mult: float = 3.0
    macd_fast: int = 8
    macd_slow: int = 21
    smooth_period: int = 3
    chandelier_mult: float = 3.0

    _st_line: np.ndarray | None = None
    _st_dir: np.ndarray | None = None
    _macd_line: np.ndarray | None = None
    _macd_signal: np.ndarray | None = None
    _macd_hist: np.ndarray | None = None
    _obv_line: np.ndarray | None = None
    _obv_ema: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._st_line, self._st_dir = supertrend(self._highs, self._lows, self._closes,
                                                  period=self.st_period, multiplier=self.st_mult)
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=5,
        )
        self._obv_line = obv(self._closes, self._volumes)
        self._obv_ema = ema(self._obv_line, 20)

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=self.smooth_period)

    def _raw_signal(self, i: int) -> float:
        d = self._st_dir[i]
        ml = self._macd_line[i]
        ov = self._obv_line[i]
        oe = self._obv_ema[i]

        if np.isnan(d) or np.isnan(ml) or np.isnan(oe):
            return 0.0

        # SuperTrend must be bearish (-1)
        if d != -1.0:
            return 0.0

        # MACD line must be negative
        if ml >= 0.0:
            return 0.0

        # OBV declining: below its own EMA
        if ov >= oe:
            return 0.0

        strength = -0.60
        if ml < -0.5:
            strength -= 0.20
        if ov < oe * 0.98:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "SuperTrend", "array": self._st_line, "type": "overlay"},
            {"name": "MACD Line", "array": self._macd_line, "panel": "MACD"},
            {"name": "MACD Signal", "array": self._macd_signal, "panel": "MACD"},
            {"name": "MACD Hist", "array": self._macd_hist, "style": "bar", "panel": "MACD",
             "color_positive": "#26a69a", "color_negative": "#ef5350"},
            {"name": "OBV", "array": self._obv_line, "panel": "OBV"},
            {"name": "OBV EMA(20)", "array": self._obv_ema, "panel": "OBV"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay("SuperTrend", datetimes, self._st_line, style="step", color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot("MACD", [
                    self._make_subplot_trace("MACD Line", datetimes, self._macd_line, color="#42a5f5"),
                    self._make_subplot_trace("MACD Signal", datetimes, self._macd_signal, color="#ff7043"),
                    self._make_subplot_trace("MACD Hist", datetimes, self._macd_hist, style="bar",
                                             color_positive="#26a69a", color_negative="#ef5350"),
                ], zero_line=True),
                self._make_subplot("OBV", [
                    self._make_subplot_trace("OBV", datetimes, self._obv_line, color="#66bb6a"),
                    self._make_subplot_trace("OBV EMA(20)", datetimes, self._obv_ema, color="#ff7043"),
                ]),
            ],
        }
