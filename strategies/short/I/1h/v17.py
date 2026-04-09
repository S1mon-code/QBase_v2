"""MildTrendShortI1hV17 — Vortex bearish + CCI oversold.

Economic logic: Vortex(12) VI- > VI+ on 1H Iron Ore captures fast rotational
bearish momentum. CCI(12) < -50 confirms the instrument is trading below its
statistical mean, reinforcing the short signal with momentum evidence.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.vortex import vortex
from indicators.trend.ema import ema
from indicators.momentum.cci import cci
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV17(TrendingStrategy):
    """Vortex(12) VI- > VI+ + CCI(12) < -50."""

    name = "short_I_1h_v17"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 25

    vortex_period: int = 12
    cci_period: int = 12
    cci_threshold: float = -50.0
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
        self._vi_plus, self._vi_minus = vortex(self._highs, self._lows, self._closes, self.vortex_period)
        self._cci = cci(self._highs, self._lows, self._closes, self.cci_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        vp = self._vi_plus[bar_index]
        vm = self._vi_minus[bar_index]
        c = self._cci[bar_index]

        if any(np.isnan(v) for v in (vp, vm, c)):
            return 0.0

        if vm <= vp:
            return 0.0

        spread = vm - vp
        signal = -(0.3 + min(0.2, spread * 2.0))

        if c < self.cci_threshold:
            signal -= min(0.15, abs(c + 50) / 200.0)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"VI+({self.vortex_period})", "array": self._vi_plus, "type": "subplot",
             "panel": "Vortex", "color": "#66bb6a"},
            {"name": f"VI-({self.vortex_period})", "array": self._vi_minus, "type": "subplot",
             "panel": "Vortex", "color": "#ef5350"},
            {"name": f"CCI({self.cci_period})", "array": self._cci, "type": "subplot",
             "horizontal_lines": [-50, 0, 50]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"Vortex({self.vortex_period})",
                    [
                        self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#66bb6a"),
                        self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                    ],
                ),
                self._make_subplot(
                    f"CCI({self.cci_period})",
                    [self._make_subplot_trace("CCI", datetimes, self._cci, color="#ab47bc")],
                    horizontal_lines=[-50, 0, 50],
                ),
            ],
        }
