"""StrongTrendShortAG2hV13 — MACD(10,24,7) bearish + Klinger(14,28) < 0.

Economic logic: MACD(10,24,7) on 2H silver detects momentum shifts at an
intermediate pace. MACD line below signal confirms bearish momentum. Klinger
Volume Oscillator negative validates that volume-weighted accumulation favors
sellers — money is flowing out during the downtrend.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.ema import ema
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV13(TrendingStrategy):
    """MACD(10,24,7) bearish + Klinger(14,28) < 0."""

    name = "strong_trend_short_AG_2h_v13"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 40

    macd_fast: int = 10
    macd_slow: int = 24
    macd_signal: int = 7
    klinger_fast: int = 14
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._macd_line, self._macd_sig, self._macd_hist = macd(
            self._closes, self.macd_fast, self.macd_slow, self.macd_signal,
        )
        self._klinger_line, _ = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=28,
        )

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        ml = self._macd_line[i]
        ms = self._macd_sig[i]
        kl = self._klinger_line[i]

        if any(np.isnan(v) for v in (ml, ms, kl)):
            return 0.0

        if ml >= ms or kl >= 0:
            return 0.0

        strength = -0.45
        spread = ms - ml
        if spread > 0.4:
            strength -= 0.25
        if kl < -500:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "MACD", "array": self._macd_line, "panel": "MACD"},
            {"name": "Signal", "array": self._macd_sig, "panel": "MACD", "color": "#ff9800"},
            {"name": "Klinger", "array": self._klinger_line, "panel": "Klinger", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot("MACD", [
                    self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#42a5f5"),
                    self._make_subplot_trace("Signal", datetimes, self._macd_sig, color="#ff9800"),
                ], zero_line=True),
                self._make_subplot("Klinger", [
                    self._make_subplot_trace("Klinger", datetimes, self._klinger_line, color="#7e57c2"),
                ], zero_line=True),
            ],
        }
