"""MildTrendShortI1hV11 — EMA bearish cross + RSI weak.

Economic logic: EMA(10) crossing below EMA(25) on 1H Iron Ore signals a
short-term bearish momentum shift. RSI(10) < 45 confirms fading buying
pressure, supporting a mild short bias without oversold extremes.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.momentum.rsi import rsi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV11(TrendingStrategy):
    """EMA(10) < EMA(25) + RSI(10) < 45."""

    name = "mild_trend_short_I_1h_v11"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 35

    ema_fast: int = 10
    ema_slow: int = 25
    rsi_period: int = 10
    chandelier_mult: float = 2.8

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

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        r = self._rsi[bar_index]

        if any(np.isnan(v) for v in (ef, es, r)):
            return 0.0

        if ef >= es:
            return 0.0

        spread = (es - ef) / es if es != 0 else 0.0
        signal = -(0.3 + min(0.2, spread * 8.0))

        if r < 45:
            signal -= min(0.15, (45 - r) / 100.0)

        return max(-0.65, signal)

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
            ],
        }
