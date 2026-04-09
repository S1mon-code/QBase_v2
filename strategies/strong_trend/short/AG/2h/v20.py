"""StrongTrendShortAG2hV20 — Fisher Transform(12) bearish + CMF(18) < 0.

Economic logic: Fisher Transform(12) on 2H silver converts prices to a Gaussian
distribution, making turning points sharper. Fisher below its signal line confirms
bearish momentum. CMF(18) negative validates distribution — selling volume
dominates at the intermediate timeframe.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.fisher_transform import fisher_transform
from indicators.trend.ema import ema
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV20(TrendingStrategy):
    """Fisher Transform(12) bearish + CMF(18) < 0."""

    name = "strong_trend_short_AG_2h_v20"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 25

    fisher_period: int = 12
    cmf_period: int = 18
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._fisher, self._fisher_signal = fisher_transform(
            self._highs, self._lows, self.fisher_period,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        fl = self._fisher[i]
        fs = self._fisher_signal[i]
        cf = self._cmf[i]

        if any(np.isnan(v) for v in (fl, fs, cf)):
            return 0.0

        if fl >= fs or cf >= 0:
            return 0.0

        strength = -0.45
        if fl < -1.0:
            strength -= 0.25
        if cf < -0.15:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "Fisher", "array": self._fisher, "panel": "Fisher"},
            {"name": "Fisher Signal", "array": self._fisher_signal, "panel": "Fisher", "color": "#ff9800"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "panel": "CMF", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"Fisher Transform({self.fisher_period})", [
                    self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#42a5f5"),
                    self._make_subplot_trace("Signal", datetimes, self._fisher_signal, color="#ff9800"),
                ], zero_line=True),
                self._make_subplot(f"CMF({self.cmf_period})", [
                    self._make_subplot_trace("CMF", datetimes, self._cmf, color="#7e57c2"),
                ], zero_line=True),
            ],
        }
