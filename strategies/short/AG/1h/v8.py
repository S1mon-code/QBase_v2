"""StrongTrendShortAG1hV8 — Donchian Breakdown + PPO Negative + Chaikin Oscillator.

Economic logic: Price below Donchian lower on 1H makes new intraday lows.
PPO < 0 confirms percentage-based momentum is bearish. Chaikin Oscillator
negative shows A/D line declining — distribution phase.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.ppo import ppo
from indicators.trend.donchian import donchian
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV8(TrendingStrategy):
    """Donchian breakdown + PPO negative + Chaikin distribution.

    Signal logic:
        close < DC_lower AND PPO_line < -0.5 AND ChaikinOsc < 0 -> -0.85
        close < DC_mid AND PPO_line < 0 -> -0.50
        else -> 0.0
    """

    name = "short_AG_1h_v8"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    dc_period: int = 25
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.dc_period,
        )
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(
            self._closes, fast=7, slow=20, signal=6,
        )
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=7, slow=22,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        dl = self._dc_lower[bar_index]
        dm = self._dc_mid[bar_index]
        p = self._ppo_line[bar_index]
        ch = self._chaikin[bar_index]

        if np.isnan(c) or np.isnan(dl) or np.isnan(p):
            return 0.0

        if c < dl and p < -0.5 and (not np.isnan(ch) and ch < 0):
            return -0.85
        if c < dm and p < 0:
            return -0.50
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "style": "dash"},
            {"name": f"DC Mid({self.dc_period})", "array": self._dc_mid, "type": "overlay"},
            {"name": f"DC Lower({self.dc_period})", "array": self._dc_lower, "type": "overlay", "style": "dash"},
            {"name": "PPO Line", "array": self._ppo_line, "panel": "PPO"},
            {"name": "PPO Signal", "array": self._ppo_signal, "panel": "PPO"},
            {"name": "ChaikinOsc", "array": self._chaikin, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"DC Mid({self.dc_period})", datetimes, self._dc_mid, color="#ffab40"),
            self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, style="dash", color="#26a69a"),
        ]
        subplots = [
            self._make_subplot("PPO", [
                self._make_subplot_trace("PPO Line", datetimes, self._ppo_line, color="#42a5f5"),
                self._make_subplot_trace("PPO Signal", datetimes, self._ppo_signal, color="#ff7043"),
            ], zero_line=True),
            self._make_subplot("ChaikinOsc", [
                self._make_subplot_trace("ChaikinOsc", datetimes, self._chaikin, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
