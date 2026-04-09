"""MildTrendShortI1hV2 — SuperTrend bearish + MACD negative + OBV declining.

Economic logic: SuperTrend(12,2.8) bearish direction confirms structural downtrend
on 1H iron ore. MACD(6,18,5) line below zero validates momentum is negative.
OBV declining below its EMA(16) shows volume leaving the market on rallies.
Signal smoothed with EMA(3) to prevent overtrading in mild-trend conditions.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.momentum.macd import macd
from indicators.volume.obv import obv
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV2(TrendingStrategy):
    """SuperTrend(12,2.8) bearish + MACD(6,18,5) < 0 + OBV declining."""

    name = "mild_trend_short_I_1h_v2"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    st_period: int = 12
    st_mult: float = 2.8
    macd_fast: int = 6
    macd_slow: int = 18
    chandelier_mult: float = 2.7

    _st_line: np.ndarray | None = None
    _st_dir: np.ndarray | None = None
    _macd_line: np.ndarray | None = None
    _macd_hist: np.ndarray | None = None
    _obv_arr: np.ndarray | None = None
    _obv_ema: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

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
        self._macd_line, _, self._macd_hist = macd(
            self._closes, self.macd_fast, self.macd_slow, 5,
        )
        self._obv_arr = obv(self._closes, self._volumes)
        self._obv_ema = ema(self._obv_arr, 16)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        d = self._st_dir[bar_index]
        ml = self._macd_line[bar_index]
        ov = self._obv_arr[bar_index]
        ov_ema = self._obv_ema[bar_index]

        if any(np.isnan(v) for v in (d, ml, ov, ov_ema)):
            return 0.0

        if d != -1:
            return 0.0

        signal = -0.35

        if ml < 0:
            signal -= min(0.15, abs(ml) * 0.01)

        if ov < ov_ema:
            signal -= 0.15

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"SuperTrend({self.st_period},{self.st_mult})", "array": self._st_line,
             "type": "overlay", "style": "step", "color": "#ff7043"},
            {"name": "MACD", "array": self._macd_line, "type": "subplot", "panel": "MACD", "zero_line": True},
            {"name": "MACD Hist", "array": self._macd_hist, "type": "subplot", "panel": "MACD",
             "style": "bar", "color_positive": "#26a69a", "color_negative": "#ef5350"},
            {"name": "OBV", "array": self._obv_arr, "type": "subplot", "panel": "OBV"},
            {"name": "OBV EMA", "array": self._obv_ema, "type": "subplot", "panel": "OBV", "color": "#ffab40"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(
                    f"SuperTrend({self.st_period},{self.st_mult})", datetimes,
                    self._st_line, style="step", color="#ff7043",
                ),
            ],
            "subplots": [
                self._make_subplot(
                    "MACD",
                    [
                        self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#bb86fc"),
                        self._make_subplot_trace("Hist", datetimes, self._macd_hist, style="bar",
                                                 color_positive="#26a69a", color_negative="#ef5350"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv_arr, color="#42a5f5"),
                        self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#ffab40"),
                    ],
                ),
            ],
        }
