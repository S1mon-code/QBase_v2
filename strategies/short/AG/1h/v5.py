"""StrongTrendShortAG1hV5 — Donchian(25) lower break + ROC(12) + Klinger(15).

Economic logic: Price breaking below the Donchian lower channel means making
new 25-bar lows -- a structural breakdown signal with built-in hysteresis
(the channel level only moves when new extremes form).  ROC < 0 confirms
negative price momentum.  Klinger oscillator < 0 validates that volume-weighted
money flow is bearish.  3-bar EMA smoothing on composite.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.trend.donchian import donchian
from indicators.trend.ema import ema
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV5(TrendingStrategy):
    """Donchian(25) lower break + ROC(12) < 0 + Klinger(15) < 0."""

    name = "short_AG_1h_v5"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    dc_period: int = 25
    roc_period: int = 12
    smooth_period: int = 4
    chandelier_mult: float = 3.0

    _dc_upper: np.ndarray | None = None
    _dc_lower: np.ndarray | None = None
    _dc_mid: np.ndarray | None = None
    _roc_line: np.ndarray | None = None
    _klinger_line: np.ndarray | None = None
    _klinger_sig: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, self.dc_period,
        )
        self._roc_line = rate_of_change(self._closes, self.roc_period)
        self._klinger_line, self._klinger_sig = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=15, slow=34, signal=13,
        )

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=self.smooth_period)

    def _raw_signal(self, i: int) -> float:
        dl = self._dc_lower[i]
        r = self._roc_line[i]
        kl = self._klinger_line[i]
        c = self._closes[i]

        if np.isnan(dl) or np.isnan(r) or np.isnan(kl):
            return 0.0

        # Price at or below Donchian lower (breakout to downside)
        if c > dl:
            return 0.0

        # ROC negative confirms momentum
        if r >= 0.0:
            return 0.0

        # Klinger oscillator negative
        if kl >= 0.0:
            return 0.0

        strength = -0.55
        if r < -2.0:
            strength -= 0.20
        if kl < -5000.0:
            strength -= 0.25
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "Donchian Upper", "array": self._dc_upper, "type": "overlay"},
            {"name": "Donchian Lower", "array": self._dc_lower, "type": "overlay"},
            {"name": "Donchian Mid", "array": self._dc_mid, "type": "overlay", "style": "dash"},
            {"name": "ROC(12)", "array": self._roc_line, "type": "subplot", "zero_line": True},
            {"name": "Klinger", "array": self._klinger_line, "panel": "Klinger", "zero_line": True},
            {"name": "Klinger Signal", "array": self._klinger_sig, "panel": "Klinger"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay("DC Upper", datetimes, self._dc_upper, style="dash", color="#66bb6a"),
                self._make_overlay("DC Lower", datetimes, self._dc_lower, style="dash", color="#ef5350"),
                self._make_overlay("DC Mid", datetimes, self._dc_mid, style="dash", color="#78909c"),
            ],
            "subplots": [
                self._make_subplot("ROC(12)", [
                    self._make_subplot_trace("ROC", datetimes, self._roc_line, color="#bb86fc"),
                ], zero_line=True),
                self._make_subplot("Klinger", [
                    self._make_subplot_trace("KVO", datetimes, self._klinger_line, color="#42a5f5"),
                    self._make_subplot_trace("Signal", datetimes, self._klinger_sig, color="#ff7043"),
                ], zero_line=True),
            ],
        }
