"""StrongTrendShortAG4hV9 — FRAMA Bearish + Vortex Bear + Chaikin Oscillator.

Economic logic: FRAMA adapts to silver's fractal volatility on 4H. Vortex VI- > VI+
confirms bearish directional movement dominates. Chaikin Oscillator < 0 validates
the A/D line is declining (distribution).
"""
from __future__ import annotations

import numpy as np

from indicators.trend.frama import frama
from indicators.trend.vortex import vortex
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV9(TrendingStrategy):
    """FRAMA bearish + Vortex bear + Chaikin distribution.

    Signal logic:
        close < FRAMA AND VI_minus > VI_plus AND ChaikinOsc < 0 -> -0.85
        close < FRAMA AND VI_minus > VI_plus -> -0.45
        else -> 0.0
    """

    name = "strong_trend_short_AG_4h_v9"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    frama_period: int = 50
    vortex_period: int = 35
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._frama = frama(self._closes, period=self.frama_period)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period,
        )
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=12, slow=35,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        f = self._frama[bar_index]
        vp = self._vi_plus[bar_index]
        vm = self._vi_minus[bar_index]
        ch = self._chaikin[bar_index]

        if np.isnan(c) or np.isnan(f) or np.isnan(vp) or np.isnan(vm):
            return 0.0

        if c >= f:
            return 0.0

        if vm > vp and (not np.isnan(ch) and ch < 0):
            return -0.85
        if vm > vp:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"FRAMA({self.frama_period})", "array": self._frama, "type": "overlay"},
            {"name": "VI+", "array": self._vi_plus, "panel": "Vortex"},
            {"name": "VI-", "array": self._vi_minus, "panel": "Vortex"},
            {"name": "ChaikinOsc", "array": self._chaikin, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"FRAMA({self.frama_period})", datetimes, self._frama, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot("Vortex", [
                self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#26a69a"),
                self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
            ], horizontal_lines=[1.0]),
            self._make_subplot("ChaikinOsc", [
                self._make_subplot_trace("ChaikinOsc", datetimes, self._chaikin, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
