"""MildTrendShortIDailyV9 — DEMA Slope + Vortex Bearish + EMV Negative.

Economic logic: DEMA(60) slope turning negative confirms the double-smoothed
trend direction is bearish. Vortex VI- exceeding VI+ validates bearish
trend energy. Negative Ease of Movement confirms that price decline is
occurring with relative ease (low volume resistance).
"""
from __future__ import annotations

import numpy as np

from indicators.trend.dema import dema
from indicators.trend.vortex import vortex
from indicators.volume.emv import emv
from indicators.trend.linear_regression import linear_regression_slope
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV9(TrendingStrategy):
    """DEMA(60) slope < 0 + Vortex(20) VI- > VI+ + EMV(18) < 0."""

    name = "short_I_daily_v9"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    # Optimizable parameters (<=5 including chandelier_mult)
    dema_period: int = 60
    vortex_period: int = 20
    emv_period: int = 18
    chandelier_mult: float = 3.5

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
        self._dema_line = dema(self._closes, self.dema_period)
        self._dema_slope = linear_regression_slope(self._dema_line, 10)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, self.vortex_period,
        )
        self._emv = emv(self._highs, self._lows, self._volumes, self.emv_period)

        # Pre-smooth raw signal
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        slope = self._dema_slope[bar_index]
        vi_p = self._vi_plus[bar_index]
        vi_m = self._vi_minus[bar_index]
        emv_val = self._emv[bar_index]

        if any(np.isnan(v) for v in (slope, vi_p, vi_m, emv_val)):
            return 0.0

        # DEMA slope must be negative
        if slope >= 0:
            return 0.0

        # Vortex: VI- must dominate
        if vi_m <= vi_p:
            return 0.0

        # Vortex spread drives base signal
        vortex_spread = vi_m - vi_p
        vortex_strength = min(vortex_spread * 5.0, 1.0)
        base = -0.3 - vortex_strength * 0.3  # -0.3 to -0.6

        # EMV confirmation
        if emv_val < 0:
            base -= 0.1

        return np.clip(base, -0.7, 0.0)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DEMA({self.dema_period})", "array": self._dema_line,
             "type": "overlay", "color": "#ffab40"},
            {"name": f"VI+({self.vortex_period})", "array": self._vi_plus,
             "type": "subplot", "panel": f"Vortex({self.vortex_period})",
             "horizontal_lines": [1.0], "color": "#4caf50"},
            {"name": f"VI-({self.vortex_period})", "array": self._vi_minus,
             "type": "subplot", "panel": f"Vortex({self.vortex_period})",
             "color": "#f44336"},
            {"name": f"EMV({self.emv_period})", "array": self._emv,
             "type": "subplot", "zero_line": True, "color": "#2196f3"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"DEMA({self.dema_period})", datetimes, self._dema_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Vortex({self.vortex_period})",
                    [
                        self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#4caf50"),
                        self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#f44336"),
                    ],
                    horizontal_lines=[1.0],
                ),
                self._make_subplot(
                    f"EMV({self.emv_period})",
                    [self._make_subplot_trace("EMV", datetimes, self._emv, color="#2196f3")],
                    zero_line=True,
                ),
            ],
        }
