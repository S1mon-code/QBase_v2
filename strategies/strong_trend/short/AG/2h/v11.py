"""StrongTrendShortAG2hV11 — EMA(15,40) bearish + RSI(12) < 44 + OBV declining.

Economic logic: EMA(15) below EMA(40) on 2H silver confirms intermediate
downtrend. RSI(12) below 44 validates weakening buying pressure. OBV below
its SMA confirms distribution — selling volume dominates the 2H timeframe.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.ema import ema
from indicators.trend.sma import sma
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV11(TrendingStrategy):
    """EMA(15) < EMA(40) + RSI(12) < 44 + OBV < SMA(30)."""

    name = "strong_trend_short_AG_2h_v11"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    ema_fast: int = 15
    ema_slow: int = 40
    rsi_period: int = 12
    obv_sma_period: int = 30
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast_line = ema(self._closes, self.ema_fast)
        self._ema_slow_line = ema(self._closes, self.ema_slow)
        self._rsi = rsi(self._closes, self.rsi_period)
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, period=self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        ef = self._ema_fast_line[i]
        es = self._ema_slow_line[i]
        r = self._rsi[i]
        ov = self._obv[i]
        os_ = self._obv_sma[i]

        if any(np.isnan(v) for v in (ef, es, r, ov, os_)):
            return 0.0

        if ef >= es or r >= 44.0 or ov >= os_:
            return 0.0

        strength = -0.40
        if r < 35.0:
            strength -= 0.25
        if ef < es * 0.99:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast_line, "type": "overlay"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow_line, "type": "overlay"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi,
             "panel": "RSI", "y_range": [0, 100], "horizontal_lines": [30, 44, 70]},
            {"name": "OBV", "array": self._obv, "panel": "OBV"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma, "panel": "OBV"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_fast})", datetimes, self._ema_fast_line, color="#ffab40"),
                self._make_overlay(f"EMA({self.ema_slow})", datetimes, self._ema_slow_line, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(f"RSI({self.rsi_period})", [
                    self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc"),
                ], horizontal_lines=[30, 44, 70], y_range=[0, 100]),
                self._make_subplot("OBV", [
                    self._make_subplot_trace("OBV", datetimes, self._obv, color="#7e57c2"),
                    self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ff9800"),
                ]),
            ],
        }
