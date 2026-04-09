"""StrongTrendShortAG1hV7 — ADX(18) > 20 + DI- > DI+ + Chaikin Osc(10,20) < 0.

Economic logic: ADX > 20 confirms a trend is present (not ranging).  DI- > DI+
is directional -- bearish movement dominates bullish.  These two together form
a hysteresis-like filter because ADX smooths heavily and DI crossovers are
inherently lagged.  Chaikin Oscillator < 0 validates that accumulation/
distribution momentum is declining.  5-bar EMA smoothing on composite.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.trend.ema import ema
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV7(TrendingStrategy):
    """ADX(18) > 20, DI- > DI+, Chaikin Osc(10,20) < 0."""

    name = "short_AG_1h_v7"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 45

    adx_period: int = 18
    chaikin_fast: int = 10
    chaikin_slow: int = 20
    smooth_period: int = 5
    chandelier_mult: float = 3.2

    _adx_line: np.ndarray | None = None
    _plus_di: np.ndarray | None = None
    _minus_di: np.ndarray | None = None
    _chaikin_line: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._adx_line, self._plus_di, self._minus_di = adx_with_di(
            self._highs, self._lows, self._closes, self.adx_period,
        )
        self._chaikin_line = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.chaikin_fast, slow=self.chaikin_slow,
        )

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=self.smooth_period)

    def _raw_signal(self, i: int) -> float:
        ax = self._adx_line[i]
        pd = self._plus_di[i]
        md = self._minus_di[i]
        ch = self._chaikin_line[i]

        if np.isnan(ax) or np.isnan(pd) or np.isnan(md) or np.isnan(ch):
            return 0.0

        # ADX confirms trend exists
        if ax <= 20.0:
            return 0.0

        # DI- must dominate DI+
        if md <= pd:
            return 0.0

        # Chaikin confirms distribution
        if ch >= 0.0:
            return 0.0

        # Graduated: stronger with wider DI gap and stronger ADX
        strength = -0.50
        if ax > 30.0:
            strength -= 0.15
        if (md - pd) > 10.0:
            strength -= 0.15
        if ch < -5000.0:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "ADX(18)", "array": self._adx_line, "panel": "ADX/DI",
             "y_range": [0, 100], "horizontal_lines": [20, 30]},
            {"name": "+DI", "array": self._plus_di, "panel": "ADX/DI"},
            {"name": "-DI", "array": self._minus_di, "panel": "ADX/DI"},
            {"name": "Chaikin Osc", "array": self._chaikin_line, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot("ADX / DI", [
                    self._make_subplot_trace("ADX", datetimes, self._adx_line, color="#ffab40"),
                    self._make_subplot_trace("+DI", datetimes, self._plus_di, color="#66bb6a"),
                    self._make_subplot_trace("-DI", datetimes, self._minus_di, color="#ef5350"),
                ], horizontal_lines=[20, 30], y_range=[0, 100]),
                self._make_subplot("Chaikin Osc", [
                    self._make_subplot_trace("Chaikin", datetimes, self._chaikin_line, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
