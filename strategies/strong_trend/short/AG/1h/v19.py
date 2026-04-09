"""StrongTrendShortAG1hV19 — TRIX(8) bearish + CMF(8) < 0 + ADX(8) > 18.

Economic logic: TRIX(8) below signal on 1H silver confirms short-term momentum
is bearish with triple smoothing. CMF(8) negative validates selling pressure.
ADX(8) above 18 ensures the move has trend strength — not just noise.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.trix import trix
from indicators.trend.adx import adx
from indicators.trend.ema import ema
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV19(TrendingStrategy):
    """TRIX(8) bearish + CMF(8) < 0 + ADX(8) > 18."""

    name = "strong_trend_short_AG_1h_v19"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    trix_period: int = 8
    cmf_period: int = 8
    adx_period: int = 8
    adx_threshold: float = 18.0
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._trix_line, self._trix_signal = trix(self._closes, self.trix_period)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)
        self._adx_line = adx(self._highs, self._lows, self._closes, self.adx_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, i: int) -> float:
        tl = self._trix_line[i]
        ts = self._trix_signal[i]
        cf = self._cmf[i]
        a = self._adx_line[i]

        if any(np.isnan(v) for v in (tl, ts, cf, a)):
            return 0.0

        if tl >= ts or cf >= 0 or a < self.adx_threshold:
            return 0.0

        strength = -0.45
        adx_str = min(1.0, (a - self.adx_threshold) / 25.0)
        strength -= adx_str * 0.30
        if cf < -0.15:
            strength -= 0.15
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "TRIX", "array": self._trix_line, "panel": "TRIX"},
            {"name": "TRIX Signal", "array": self._trix_signal, "panel": "TRIX", "color": "#ff9800"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "panel": "CMF", "zero_line": True},
            {"name": f"ADX({self.adx_period})", "array": self._adx_line,
             "panel": "ADX", "y_range": [0, 100], "horizontal_lines": [18]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"TRIX({self.trix_period})", [
                    self._make_subplot_trace("TRIX", datetimes, self._trix_line, color="#42a5f5"),
                    self._make_subplot_trace("Signal", datetimes, self._trix_signal, color="#ff9800"),
                ], zero_line=True),
                self._make_subplot(f"CMF({self.cmf_period})", [
                    self._make_subplot_trace("CMF", datetimes, self._cmf, color="#7e57c2"),
                ], zero_line=True),
                self._make_subplot(f"ADX({self.adx_period})", [
                    self._make_subplot_trace("ADX", datetimes, self._adx_line, color="#66bb6a"),
                ], horizontal_lines=[18], y_range=[0, 100]),
            ],
        }
