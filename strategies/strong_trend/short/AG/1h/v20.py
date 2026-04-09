"""StrongTrendShortAG1hV20 — TSI(8,16) bearish + OBV below SMA(12).

Economic logic: TSI(8,16) on 1H silver provides a double-smoothed momentum
reading. TSI below its signal confirms bearish momentum. OBV below SMA(12)
validates distribution — volume-weighted selling persists over the short term.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.trend.ema import ema
from indicators.trend.sma import sma
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV20(TrendingStrategy):
    """TSI(8,16) bearish + OBV < SMA(12)."""

    name = "strong_trend_short_AG_1h_v20"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 25

    tsi_fast: int = 8
    tsi_slow: int = 16
    obv_sma_period: int = 12
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tsi_line, self._tsi_signal = tsi(self._closes, long_period=self.tsi_slow, short_period=self.tsi_fast)
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, period=self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, i: int) -> float:
        tl = self._tsi_line[i]
        ts = self._tsi_signal[i]
        ov = self._obv[i]
        os_ = self._obv_sma[i]

        if any(np.isnan(v) for v in (tl, ts, ov, os_)):
            return 0.0

        if tl >= ts or ov >= os_:
            return 0.0

        strength = -0.45
        if tl < -10.0:
            strength -= 0.25
        if tl < -20.0:
            strength -= 0.15
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "TSI", "array": self._tsi_line, "panel": "TSI"},
            {"name": "TSI Signal", "array": self._tsi_signal, "panel": "TSI", "color": "#ff9800"},
            {"name": "OBV", "array": self._obv, "panel": "OBV"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma, "panel": "OBV"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"TSI({self.tsi_fast},{self.tsi_slow})", [
                    self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#42a5f5"),
                    self._make_subplot_trace("Signal", datetimes, self._tsi_signal, color="#ff9800"),
                ], zero_line=True),
                self._make_subplot("OBV", [
                    self._make_subplot_trace("OBV", datetimes, self._obv, color="#7e57c2"),
                    self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ffab40"),
                ]),
            ],
        }
