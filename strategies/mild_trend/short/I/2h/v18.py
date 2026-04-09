"""MildTrendShortI2hV18 — HMA slope negative + RSI weak.

Economic logic: HMA(35) declining slope on 2H Iron Ore provides smooth,
lag-reduced bearish momentum detection. RSI(14) < 45 confirms fading upward
pressure, supporting the short bias without oversold extremes.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.trend.ema import ema
from indicators.momentum.rsi import rsi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV18(TrendingStrategy):
    """HMA(35) slope < 0 + RSI(14) < 45."""

    name = "mild_trend_short_I_2h_v18"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 50

    hma_period: int = 35
    rsi_period: int = 14
    chandelier_mult: float = 3.0

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
        self._hma = hma(self._closes, self.hma_period)
        self._rsi = rsi(self._closes, self.rsi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1]
        r = self._rsi[bar_index]

        if any(np.isnan(v) for v in (h, h_prev, r)):
            return 0.0

        slope = h - h_prev
        if slope >= 0:
            return 0.0

        slope_pct = abs(slope) / h_prev if h_prev != 0 else 0.0
        signal = -(0.3 + min(0.2, slope_pct * 250))

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
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay", "color": "#ff7043"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 45, 70]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    f"RSI({self.rsi_period})",
                    [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc")],
                    horizontal_lines=[30, 45, 70], y_range=[0, 100],
                ),
            ],
        }
