"""StrongTrendShortAGDailyV8 — TEMA Bearish + ROC Negative + Chaikin Oscillator.

Economic logic: TEMA gives fast trend response — price below TEMA120 confirms
daily downtrend. ROC < 0 validates negative momentum. Chaikin Oscillator < 0
shows accumulation/distribution line is declining (distribution phase).
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.trend.tema import tema
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV8(TrendingStrategy):
    """TEMA bearish + ROC negative + Chaikin distribution.

    Signal logic:
        close < TEMA AND ROC < -2 AND ChaikinOsc < 0 -> -0.85
        close < TEMA AND ROC < 0 -> -0.45
        else -> 0.0
    """

    name = "short_AG_daily_v8"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 200

    tema_period: int = 120
    roc_period: int = 80
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, self.tema_period)
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=30, slow=80,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        t = self._tema[bar_index]
        r = self._roc[bar_index]
        ch = self._chaikin[bar_index]

        if np.isnan(c) or np.isnan(t) or np.isnan(r):
            return 0.0

        if c >= t:
            return 0.0

        if r < -2 and (not np.isnan(ch) and ch < 0):
            return -0.85
        if r < 0:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay"},
            {"name": f"ROC({self.roc_period})", "array": self._roc, "zero_line": True},
            {"name": "ChaikinOsc", "array": self._chaikin, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot(f"ROC({self.roc_period})", [
                self._make_subplot_trace("ROC", datetimes, self._roc, color="#42a5f5"),
            ], zero_line=True),
            self._make_subplot("ChaikinOsc", [
                self._make_subplot_trace("ChaikinOsc", datetimes, self._chaikin, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
