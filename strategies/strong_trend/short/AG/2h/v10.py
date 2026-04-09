"""StrongTrendShortAG2hV10 — DEMA Slope + Coppock Negative + OBV Below SMA.

Economic logic: DEMA(35) slope < 0 provides double-smoothed trend direction
with reduced lag. Coppock curve < 0 confirms long-term momentum is bearish.
OBV below its SMA(30) validates sustained distribution. EMA(4) smoothing
reduces overtrading in noisy silver downtrends.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.dema import dema
from indicators.trend.sma import sma
from indicators.momentum.coppock import coppock
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV10(TrendingStrategy):
    """DEMA(35) slope < 0 + Coppock < 0 + OBV < OBV_SMA(30).

    Signal logic (raw, pre-smoothing):
        All 3 conditions met -> -0.85
        DEMA slope < 0 AND Coppock < 0 -> -0.55
        DEMA slope < 0 AND OBV < OBV_SMA -> -0.35
        else -> 0.0
    """

    name = "strong_trend_short_AG_2h_v10"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 65

    dema_period: int = 35
    obv_sma_period: int = 30
    chandelier_mult: float = 3.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dema = dema(self._closes, period=self.dema_period)
        self._coppock = coppock(self._closes, wma_period=10, roc_long=14,
                                roc_short=11)
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, period=self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        d = self._dema[i]
        d_prev = self._dema[i - 1] if i > 0 else np.nan
        cp = self._coppock[i]
        ov = self._obv[i]
        os = self._obv_sma[i]

        if any(np.isnan(v) for v in (d, d_prev, cp, ov, os)):
            return 0.0

        dema_slope_neg = d < d_prev
        coppock_neg = cp < 0.0
        obv_below_sma = ov < os

        if dema_slope_neg and coppock_neg and obv_below_sma:
            return -0.85
        if dema_slope_neg and coppock_neg:
            return -0.55
        if dema_slope_neg and obv_below_sma:
            return -0.35
        return 0.0

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DEMA({self.dema_period})", "array": self._dema,
             "type": "overlay"},
            {"name": "Coppock", "array": self._coppock, "zero_line": True},
            {"name": "OBV", "array": self._obv, "panel": "OBV"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma,
             "panel": "OBV"},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"DEMA({self.dema_period})", datetimes,
                               self._dema, color="#ffab40"),
        ]
        subplots = [
            self._make_subplot("Coppock", [
                self._make_subplot_trace("Coppock", datetimes, self._coppock,
                                         color="#42a5f5"),
            ], zero_line=True),
            self._make_subplot("OBV", [
                self._make_subplot_trace("OBV", datetimes, self._obv,
                                         color="#66bb6a"),
                self._make_subplot_trace(f"OBV SMA({self.obv_sma_period})", datetimes,
                                         self._obv_sma, color="#ef5350"),
            ]),
        ]
        return {"overlays": overlays, "subplots": subplots}
