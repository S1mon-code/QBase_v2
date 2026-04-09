"""StrongTrendShortAG1hV11 — EMA(8,22) bearish + RSI(8) < 42 + Volume spike(8).

Economic logic: Fast EMA(8) below EMA(22) on 1H silver confirms short-term
downtrend. RSI(8) < 42 validates weakening buying pressure. Volume spike
detects institutional selling — sudden volume bursts during downtrends signal
large players liquidating positions.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.ema import ema
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV11(TrendingStrategy):
    """EMA(8) < EMA(22) + RSI(8) < 42 + Volume spike(8)."""

    name = "strong_trend_short_AG_1h_v11"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    ema_fast: int = 8
    ema_slow: int = 22
    rsi_period: int = 8
    vs_period: int = 8
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast_line = ema(self._closes, self.ema_fast)
        self._ema_slow_line = ema(self._closes, self.ema_slow)
        self._rsi = rsi(self._closes, self.rsi_period)
        self._vs = volume_spike(self._volumes, self.vs_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, i: int) -> float:
        ef = self._ema_fast_line[i]
        es = self._ema_slow_line[i]
        r = self._rsi[i]
        vs = self._vs[i]

        if any(np.isnan(v) for v in (ef, es, r, vs)):
            return 0.0

        if ef >= es or r >= 42.0:
            return 0.0

        strength = -0.40
        if vs > 1.5:
            strength -= 0.25
        if r < 30.0:
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
             "panel": "RSI", "y_range": [0, 100], "horizontal_lines": [30, 42, 70]},
            {"name": f"Vol Spike({self.vs_period})", "array": self._vs, "panel": "VS"},
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
                ], horizontal_lines=[30, 42, 70], y_range=[0, 100]),
                self._make_subplot(f"Volume Spike({self.vs_period})", [
                    self._make_subplot_trace("Spike", datetimes, self._vs, color="#42a5f5"),
                ], horizontal_lines=[1.5]),
            ],
        }
