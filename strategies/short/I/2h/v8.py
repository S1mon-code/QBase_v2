"""MildTrendShortI2hV8 — Donchian Lower Break + Momentum Accel Negative + Twiggs.

Economic logic: Breaking below Donchian lower band on 2H signals new period lows.
Negative momentum acceleration (2nd derivative) confirms the decline is
accelerating. Twiggs Money Flow below zero validates persistent selling.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.momentum.momentum_accel import momentum_acceleration
from indicators.volume.twiggs import twiggs_money_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV8(TrendingStrategy):
    """Break Donchian(50) lower + MomAccel(30)<0 + Twiggs(40)<0."""

    name = "short_I_2h_v8"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 65

    dc_period: int = 50
    ma_fast: int = 30
    twiggs_period: int = 40
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
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, self.dc_period,
        )
        self._mom_accel = momentum_acceleration(self._closes, fast_period=self.ma_fast)
        self._twiggs = twiggs_money_flow(
            self._highs, self._lows, self._closes, self._volumes, self.twiggs_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_low = self._dc_lower[bar_index]
        dc_mid = self._dc_mid[bar_index]
        ma_val = self._mom_accel[bar_index]
        tw = self._twiggs[bar_index]

        if any(np.isnan(v) for v in (close, dc_low)):
            return 0.0

        signal = 0.0

        if close < dc_low:
            signal = -0.4
        elif not np.isnan(dc_mid) and close < dc_mid:
            signal = -0.2
        else:
            return 0.0

        if not np.isnan(ma_val) and ma_val < 0:
            signal -= 0.2

        if not np.isnan(tw) and tw < 0:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "style": "dash", "color": "#ef5350"},
            {"name": f"DC Lower({self.dc_period})", "array": self._dc_lower, "type": "overlay", "style": "dash", "color": "#26a69a"},
            {"name": "Mom Accel", "array": self._mom_accel, "type": "subplot", "zero_line": True},
            {"name": f"Twiggs({self.twiggs_period})", "array": self._twiggs, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, style="dash", color="#ef5350"),
                self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, style="dash", color="#26a69a"),
            ],
            "subplots": [
                self._make_subplot(
                    "Mom Accel",
                    [self._make_subplot_trace("MA", datetimes, self._mom_accel, color="#bb86fc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"Twiggs({self.twiggs_period})",
                    [self._make_subplot_trace("Twiggs", datetimes, self._twiggs, color="#ef5350")],
                    zero_line=True,
                ),
            ],
        }
