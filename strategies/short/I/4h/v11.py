"""MildTrendShortI4hV11 — EMA bearish cross + RSI weak + OBV declining.

Economic logic: EMA(30) crossing below EMA(70) on 4H Iron Ore signals a
medium-slow bearish momentum shift. RSI(16) < 45 confirms fading buying
pressure. OBV declining below its SMA validates sustained distribution.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.sma import sma
from indicators.momentum.rsi import rsi
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV11(TrendingStrategy):
    """EMA(30) < EMA(70) + RSI(16) < 45 + OBV < OBV_SMA(40)."""

    name = "short_I_4h_v11"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 90

    ema_fast: int = 30
    ema_slow: int = 70
    rsi_period: int = 16
    obv_sma_period: int = 40
    chandelier_mult: float = 3.4

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
        self._ema_fast = ema(self._closes, self.ema_fast)
        self._ema_slow = ema(self._closes, self.ema_slow)
        self._rsi = rsi(self._closes, self.rsi_period)
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, bar_index: int) -> float:
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        r = self._rsi[bar_index]
        ov = self._obv[bar_index]
        os_ = self._obv_sma[bar_index]

        if any(np.isnan(v) for v in (ef, es, r, os_)):
            return 0.0

        if ef >= es:
            return 0.0

        spread = (es - ef) / es if es != 0 else 0.0
        signal = -(0.25 + min(0.2, spread * 5.5))

        if r < 45:
            signal -= min(0.15, (45 - r) / 100.0)

        if ov < os_:
            signal -= 0.1

        return max(-0.7, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast, "type": "overlay", "color": "#ff7043"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow, "type": "overlay", "color": "#42a5f5"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 45, 70]},
            {"name": "OBV", "array": self._obv, "type": "subplot"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma,
             "type": "subplot", "panel": "OBV", "color": "#ff9800"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_fast})", datetimes, self._ema_fast, color="#ff7043"),
                self._make_overlay(f"EMA({self.ema_slow})", datetimes, self._ema_slow, color="#42a5f5"),
            ],
            "subplots": [
                self._make_subplot(
                    f"RSI({self.rsi_period})",
                    [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc")],
                    horizontal_lines=[30, 45, 70], y_range=[0, 100],
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv, color="#2196f3"),
                        self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ff9800"),
                    ],
                ),
            ],
        }
