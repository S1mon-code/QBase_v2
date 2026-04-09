"""StrongTrendShortAG1hV3 — TEMA(18) slope + Aroon Down(20) + Force Index(10).

Economic logic: TEMA negative slope detects declining price with minimal lag.
Aroon Down > 60 confirms recent lows dominate the lookback window.
Force Index < 0 validates that selling volume pressure exceeds buying.
All three filters must agree, then 3-bar EMA smoothing prevents flicker.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi  # noqa: F401 (unused, kept for reference)
from indicators.trend.aroon import aroon
from indicators.trend.ema import ema
from indicators.trend.tema import tema
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV3(TrendingStrategy):
    """TEMA(18) negative slope + Aroon Down(20) > 60 + Force Index(10) < 0."""

    name = "short_AG_1h_v3"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    tema_period: int = 18
    aroon_period: int = 20
    fi_period: int = 10
    smooth_period: int = 3
    chandelier_mult: float = 3.0

    _tema_line: np.ndarray | None = None
    _aroon_up: np.ndarray | None = None
    _aroon_down: np.ndarray | None = None
    _fi_line: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._tema_line = tema(self._closes, self.tema_period)
        self._aroon_up, self._aroon_down, _ = aroon(self._highs, self._lows, self.aroon_period)
        self._fi_line = force_index(self._closes, self._volumes, self.fi_period)

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=self.smooth_period)

    def _raw_signal(self, i: int) -> float:
        t = self._tema_line
        ad = self._aroon_down
        fi = self._fi_line

        if i < 1 or np.isnan(t[i]) or np.isnan(t[i - 1]) or np.isnan(ad[i]) or np.isnan(fi[i]):
            return 0.0

        # TEMA slope negative
        slope = t[i] - t[i - 1]
        if slope >= 0.0:
            return 0.0

        # Aroon Down > 60 (recent lows dominating)
        if ad[i] <= 60.0:
            return 0.0

        # Force Index negative (selling pressure)
        if fi[i] >= 0.0:
            return 0.0

        strength = -0.55
        if ad[i] > 80.0:
            strength -= 0.20
        if fi[i] < -1000.0:
            strength -= 0.25
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema_line, "type": "overlay"},
            {"name": "Aroon Down", "array": self._aroon_down, "panel": "Aroon",
             "y_range": [0, 100], "horizontal_lines": [60, 80]},
            {"name": "Force Index", "array": self._fi_line, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot("Aroon", [
                    self._make_subplot_trace("Aroon Down", datetimes, self._aroon_down, color="#ef5350"),
                    self._make_subplot_trace("Aroon Up", datetimes, self._aroon_up, color="#66bb6a"),
                ], horizontal_lines=[60, 80], y_range=[0, 100]),
                self._make_subplot("Force Index", [
                    self._make_subplot_trace("FI(10)", datetimes, self._fi_line, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
