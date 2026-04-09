"""MildTrendShortI4hV5 — Linear Regression Breakdown + Vortex Bearish + Chaikin Osc.

Economic logic: Price below the linear regression line on 4H signals deviation
from the fitted trend. Vortex VI- exceeding VI+ confirms bearish trend energy.
Negative Chaikin Oscillator validates distribution pressure.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression
from indicators.trend.vortex import vortex
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV5(TrendingStrategy):
    """Price below LinReg(60) + VI->VI+ + ChaikinOsc(15,40)<0."""

    name = "short_I_4h_v5"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    lr_period: int = 60
    vortex_period: int = 40
    co_fast: int = 15
    co_slow: int = 40
    chandelier_mult: float = 2.5

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
        self._lr = linear_regression(self._closes, self.lr_period)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, self.vortex_period,
        )
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            self.co_fast, self.co_slow,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        lr_val = self._lr[bar_index]
        vi_p = self._vi_plus[bar_index]
        vi_m = self._vi_minus[bar_index]
        co = self._chaikin[bar_index]

        if any(np.isnan(v) for v in (close, lr_val, vi_p, vi_m)):
            return 0.0

        if close >= lr_val:
            return 0.0

        dist = (lr_val - close) / lr_val
        strength = min(1.0, dist * 30.0)

        signal = -(0.25 + strength * 0.3)

        if vi_m > vi_p:
            signal -= 0.2

        if not np.isnan(co) and co < 0:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LinReg({self.lr_period})", "array": self._lr, "type": "overlay", "color": "#ffab40"},
            {"name": "VI+", "array": self._vi_plus, "type": "subplot", "panel": "Vortex"},
            {"name": "VI-", "array": self._vi_minus, "type": "subplot", "panel": "Vortex"},
            {"name": "Chaikin Osc", "array": self._chaikin, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"LinReg({self.lr_period})", datetimes, self._lr, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Vortex",
                    [
                        self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#26a69a"),
                        self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                    ],
                    horizontal_lines=[1.0],
                ),
                self._make_subplot(
                    "Chaikin Osc",
                    [self._make_subplot_trace("CO", datetimes, self._chaikin, color="#bb86fc")],
                    zero_line=True,
                ),
            ],
        }
