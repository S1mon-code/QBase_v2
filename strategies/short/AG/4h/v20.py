"""StrongTrendShortAG4hV20 — TRIX(14) bearish + Klinger(16,32) < 0.

Economic logic: TRIX(14) below its signal on 4H silver confirms bearish momentum
via triple exponential smoothing — highly noise-resistant. Klinger(16,32) negative
validates that volume-weighted accumulation favors sellers. The combination
identifies high-conviction intermediate-term short opportunities.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.trix import trix
from indicators.trend.ema import ema
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV20(TrendingStrategy):
    """TRIX(14) bearish + Klinger(16,32) < 0."""

    name = "short_AG_4h_v20"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    trix_period: int = 14
    klinger_fast: int = 16
    klinger_slow: int = 32
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._trix_line, self._trix_signal = trix(self._closes, self.trix_period)
        self._klinger_line, _ = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=self.klinger_slow,
        )

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        tl = self._trix_line[i]
        ts = self._trix_signal[i]
        kl = self._klinger_line[i]

        if any(np.isnan(v) for v in (tl, ts, kl)):
            return 0.0

        if tl >= ts or kl >= 0:
            return 0.0

        strength = -0.45
        spread = ts - tl
        if spread > 0.01:
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
            {"name": "TRIX", "array": self._trix_line, "panel": "TRIX"},
            {"name": "TRIX Signal", "array": self._trix_signal, "panel": "TRIX", "color": "#ff9800"},
            {"name": "Klinger", "array": self._klinger_line, "panel": "Klinger", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"TRIX({self.trix_period})", [
                    self._make_subplot_trace("TRIX", datetimes, self._trix_line, color="#42a5f5"),
                    self._make_subplot_trace("Signal", datetimes, self._trix_signal, color="#ff9800"),
                ], zero_line=True),
                self._make_subplot(f"Klinger({self.klinger_fast},{self.klinger_slow})", [
                    self._make_subplot_trace("Klinger", datetimes, self._klinger_line, color="#7e57c2"),
                ], zero_line=True),
            ],
        }
