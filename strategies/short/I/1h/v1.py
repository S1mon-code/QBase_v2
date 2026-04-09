"""MildTrendShortI1hV1 — EMA slope + RSI weak + CMF distribution.

Economic logic: EMA(18) declining slope confirms mild downtrend on 1H iron ore.
RSI(12) < 45 shows momentum fading without being oversold. CMF(12) < 0 validates
selling pressure from institutional distribution. Signal smoothed with EMA(3)
to prevent overtrading in choppy mild-trend conditions.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.momentum.rsi import rsi
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV1(TrendingStrategy):
    """EMA(18) slope < 0 + RSI(12) < 45 + CMF(12) < 0."""

    name = "short_I_1h_v1"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    ema_period: int = 18
    rsi_period: int = 12
    cmf_period: int = 12
    chandelier_mult: float = 2.8

    # --- precomputed arrays ---
    _ema_line: np.ndarray | None = None
    _rsi_arr: np.ndarray | None = None
    _cmf_arr: np.ndarray | None = None
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
        self._ema_line = ema(self._closes, self.ema_period)
        self._rsi_arr = rsi(self._closes, self.rsi_period)
        self._cmf_arr = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        e = self._ema_line[bar_index]
        e_prev = self._ema_line[bar_index - 1]
        r = self._rsi_arr[bar_index]
        c = self._cmf_arr[bar_index]

        if any(np.isnan(v) for v in (e, e_prev, r, c)):
            return 0.0

        slope = e - e_prev
        if slope >= 0:
            return 0.0

        signal = -0.3

        if r < 45:
            signal -= min(0.2, (45 - r) / 100.0)

        if c < 0:
            signal -= min(0.15, abs(c) * 0.5)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"EMA({self.ema_period})", "array": self._ema_line, "type": "overlay", "color": "#ffab40"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi_arr, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 45, 70]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf_arr, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_period})", datetimes, self._ema_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"RSI({self.rsi_period})",
                    [self._make_subplot_trace("RSI", datetimes, self._rsi_arr, color="#bb86fc")],
                    horizontal_lines=[30, 45, 70], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf_arr, color="#26c6da")],
                    zero_line=True,
                ),
            ],
        }
