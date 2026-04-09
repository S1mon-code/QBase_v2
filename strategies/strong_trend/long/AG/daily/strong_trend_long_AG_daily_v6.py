"""StrongTrendLongAGDailyV6 — LinearReg(200) + Vortex(90) + Klinger(80,120,30).

Economic logic: Linear Regression channel on wide window captures Silver's
long-term macro slope. Vortex Indicator separates positive and negative trend
movements. Klinger Volume Oscillator (substituted for SmartMoney) measures
accumulation/distribution through volume force, ideal for AG's institutional flows.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression
from indicators.trend.vortex import vortex
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAGDailyV6(TrendingStrategy):
    """Linear Regression slope + Vortex trend + Klinger volume.

    Signal logic:
        - LinReg rising AND VI+ > VI- AND Klinger > signal → long
        - Strength scales with Vortex differential

    Attributes:
        linreg_period:   Linear Regression lookback.
        vortex_period:   Vortex Indicator period.
        klinger_fast:    Klinger fast EMA.
        klinger_slow:    Klinger slow EMA.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "strong_trend_long_AG_daily_v6"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 250

    linreg_period: int = 200
    vortex_period: int = 90
    klinger_fast: int = 80
    klinger_slow: int = 120
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
        self._linreg = linear_regression(self._closes, period=self.linreg_period)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period,
        )
        self._klinger_line, self._klinger_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=self.klinger_slow, signal=30,
        )

    def _generate_signal(self, bar_index: int) -> float:
        lr = self._linreg[bar_index]
        lr_prev = self._linreg[bar_index - 1]
        vip = self._vi_plus[bar_index]
        vim = self._vi_minus[bar_index]
        kl = self._klinger_line[bar_index]
        ks = self._klinger_signal[bar_index]

        if any(np.isnan(v) for v in (lr, lr_prev, vip, vim, kl, ks)):
            return 0.0

        lr_rising = lr > lr_prev
        vortex_bull = vip > vim
        klinger_bull = kl > ks

        if lr_rising and vortex_bull and klinger_bull:
            diff = vip - vim
            strength = min(1.0, diff * 2.0 + 0.3)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LinReg({self.linreg_period})", "array": self._linreg, "type": "overlay"},
            {"name": f"VI+({self.vortex_period})", "array": self._vi_plus, "type": "subplot",
             "panel": "Vortex"},
            {"name": f"VI-({self.vortex_period})", "array": self._vi_minus, "type": "subplot",
             "panel": "Vortex", "horizontal_lines": [1.0]},
            {"name": "Klinger", "array": self._klinger_line, "type": "subplot",
             "panel": "Klinger", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"LinReg({self.linreg_period})", datetimes, self._linreg, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Vortex({self.vortex_period})",
                    [self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#66bb6a"),
                     self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350")],
                    horizontal_lines=[1.0],
                ),
                self._make_subplot(
                    "Klinger",
                    [self._make_subplot_trace("Klinger", datetimes, self._klinger_line, color="#42a5f5"),
                     self._make_subplot_trace("Signal", datetimes, self._klinger_signal, color="#ff8a65")],
                    zero_line=True,
                ),
            ],
        }
