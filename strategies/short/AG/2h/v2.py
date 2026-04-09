"""StrongTrendShortAG2hV2 — SuperTrend Bearish + MACD Histogram + OBV Declining.

Economic logic: SuperTrend(20,3.2) bearish direction confirms price below
dynamic support. MACD(10,26,8) histogram < 0 shows momentum decelerating.
OBV declining validates volume-side distribution. EMA(4) signal smoothing
reduces overtrading in choppy downtrends.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.supertrend import supertrend
from indicators.momentum.macd import macd
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV2(TrendingStrategy):
    """SuperTrend(20,3.2) bearish + MACD(10,26,8) hist < 0 + OBV declining.

    Signal logic (raw, pre-smoothing):
        All 3 conditions met -> -0.85
        SuperTrend bearish AND MACD hist < 0 -> -0.50
        SuperTrend bearish AND OBV declining -> -0.35
        else -> 0.0
    """

    name = "short_AG_2h_v2"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    st_period: int = 20
    st_mult: float = 3.2
    macd_fast: int = 10
    macd_slow: int = 26
    macd_signal: int = 8
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._macd_line, self._macd_sig, self._macd_hist = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow,
            signal=self.macd_signal,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_ema = ema(self._obv, period=10)

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        st_d = self._st_dir[i]
        hist = self._macd_hist[i]
        o = self._obv_ema[i]
        o_prev = self._obv_ema[i - 1] if i > 0 else np.nan

        if any(np.isnan(v) for v in (st_d, hist, o, o_prev)):
            return 0.0

        st_bearish = st_d < 0
        macd_neg = hist < 0.0
        obv_declining = o < o_prev

        if st_bearish and macd_neg and obv_declining:
            return -0.85
        if st_bearish and macd_neg:
            return -0.50
        if st_bearish and obv_declining:
            return -0.35
        return 0.0

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st_line,
             "type": "overlay", "style": "step"},
            {"name": "MACD Hist", "array": self._macd_hist, "panel": "MACD",
             "style": "bar", "zero_line": True},
            {"name": "MACD Line", "array": self._macd_line, "panel": "MACD"},
            {"name": "MACD Signal", "array": self._macd_sig, "panel": "MACD"},
            {"name": "OBV EMA(10)", "array": self._obv_ema, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"SuperTrend({self.st_period})", datetimes,
                               self._st_line, style="step", color="#ef5350"),
        ]
        subplots = [
            self._make_subplot("MACD", [
                self._make_subplot_trace("MACD Line", datetimes, self._macd_line,
                                         color="#42a5f5"),
                self._make_subplot_trace("Signal", datetimes, self._macd_sig,
                                         color="#ffab40"),
                self._make_subplot_trace("Histogram", datetimes, self._macd_hist,
                                         style="bar", color_positive="#66bb6a",
                                         color_negative="#ef5350"),
            ], zero_line=True),
            self._make_subplot("OBV", [
                self._make_subplot_trace("OBV EMA(10)", datetimes, self._obv_ema,
                                         color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
