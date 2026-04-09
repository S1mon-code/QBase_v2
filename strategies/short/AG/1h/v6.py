"""StrongTrendShortAG1hV6 — KAMA(20) slope + Fisher Transform(15) + CMF(15).

Economic logic: KAMA adapts its smoothing to market efficiency -- in a strong
trend it tracks closely, in chop it flattens out.  Negative KAMA slope is a
robust trend-direction signal.  Fisher Transform < -0.5 amplifies extreme
bearish price positioning.  CMF < 0 confirms distribution pressure.
4-bar EMA smoothing for extra noise suppression on 1H.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.fisher_transform import fisher_transform
from indicators.trend.ema import ema
from indicators.trend.kama import kama
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV6(TrendingStrategy):
    """KAMA(20) negative slope + Fisher(15) < -0.5 + CMF(15) < 0."""

    name = "short_AG_1h_v6"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    kama_period: int = 20
    fisher_period: int = 15
    cmf_period: int = 15
    smooth_period: int = 4
    chandelier_mult: float = 3.0

    _kama_line: np.ndarray | None = None
    _fisher_line: np.ndarray | None = None
    _fisher_trigger: np.ndarray | None = None
    _cmf_line: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._kama_line = kama(self._closes, self.kama_period)
        self._fisher_line, self._fisher_trigger = fisher_transform(
            self._highs, self._lows, self.fisher_period,
        )
        self._cmf_line = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=self.smooth_period)

    def _raw_signal(self, i: int) -> float:
        k = self._kama_line
        f = self._fisher_line
        c = self._cmf_line

        if i < 1 or np.isnan(k[i]) or np.isnan(k[i - 1]) or np.isnan(f[i]) or np.isnan(c[i]):
            return 0.0

        # KAMA slope negative
        slope = k[i] - k[i - 1]
        if slope >= 0.0:
            return 0.0

        # Fisher Transform below threshold
        if f[i] >= -0.5:
            return 0.0

        # CMF confirms distribution
        if c[i] >= 0.0:
            return 0.0

        strength = -0.55
        if f[i] < -1.5:
            strength -= 0.20
        if c[i] < -0.15:
            strength -= 0.25
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama_line, "type": "overlay"},
            {"name": "Fisher(15)", "array": self._fisher_line, "panel": "Fisher",
             "zero_line": True, "horizontal_lines": [-0.5, -1.5]},
            {"name": "Fisher Trigger", "array": self._fisher_trigger, "panel": "Fisher"},
            {"name": "CMF(15)", "array": self._cmf_line, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot("Fisher(15)", [
                    self._make_subplot_trace("Fisher", datetimes, self._fisher_line, color="#bb86fc"),
                    self._make_subplot_trace("Trigger", datetimes, self._fisher_trigger, color="#78909c"),
                ], zero_line=True, horizontal_lines=[-0.5, -1.5]),
                self._make_subplot("CMF(15)", [
                    self._make_subplot_trace("CMF", datetimes, self._cmf_line, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
