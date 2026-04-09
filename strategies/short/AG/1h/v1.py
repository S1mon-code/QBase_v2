"""StrongTrendShortAG1hV1 — EMA(20) slope + RSI(14) + CMF(15).

Economic logic: EMA slope measures the *rate* of price decline rather than
a cross, avoiding whipsaw in choppy downtrends.  RSI below 45 confirms
weakening buying pressure.  CMF < 0 validates distribution (selling volume
dominates).  Signal is EMA-smoothed (period 3) to suppress bar-to-bar noise
that plagues 1H data.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.ema import ema
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV1(TrendingStrategy):
    """EMA(20) negative slope + RSI(14) < 45 + CMF(15) < 0.

    Anti-whipsaw design: slope-based trend detection + 3-bar EMA smoothing
    on the composite signal prevents rapid signal flipping.
    """

    name = "short_AG_1h_v1"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    ema_period: int = 20
    rsi_period: int = 14
    cmf_period: int = 15
    smooth_period: int = 3
    chandelier_mult: float = 3.0

    # -- pre-computed arrays --
    _ema_line: np.ndarray | None = None
    _rsi_line: np.ndarray | None = None
    _cmf_line: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._ema_line = ema(self._closes, self.ema_period)
        self._rsi_line = rsi(self._closes, self.rsi_period)
        self._cmf_line = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        # -- build raw signal, then smooth --
        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=self.smooth_period)

    def _raw_signal(self, i: int) -> float:
        e = self._ema_line
        r = self._rsi_line
        c = self._cmf_line

        if i < 1 or np.isnan(e[i]) or np.isnan(e[i - 1]) or np.isnan(r[i]) or np.isnan(c[i]):
            return 0.0

        # Slope: negative means price declining
        slope = e[i] - e[i - 1]
        if slope >= 0.0:
            return 0.0

        # RSI confirms weakness
        if r[i] >= 45.0:
            return 0.0

        # CMF confirms distribution
        if c[i] >= 0.0:
            return 0.0

        # Graduated signal: stronger when all three are deeper
        strength = -0.50
        if r[i] < 35.0:
            strength -= 0.20
        if c[i] < -0.10:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"EMA({self.ema_period})", "array": self._ema_line, "type": "overlay"},
            {"name": "RSI(14)", "array": self._rsi_line, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 45, 70]},
            {"name": "CMF(15)", "array": self._cmf_line, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_period})", datetimes, self._ema_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot("RSI(14)", [
                    self._make_subplot_trace("RSI", datetimes, self._rsi_line, color="#bb86fc"),
                ], horizontal_lines=[30, 45, 70], y_range=[0, 100]),
                self._make_subplot("CMF(15)", [
                    self._make_subplot_trace("CMF", datetimes, self._cmf_line, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
