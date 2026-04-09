"""MildTrendShortI1hV5 — Donchian lower break + ROC negative + Klinger bearish.

Economic logic: Breaking below Donchian(20) lower channel on 1H signals new
period lows in iron ore. ROC(10) < 0 confirms price is declining versus recent
history. Klinger(12) oscillator below zero validates volume-weighted distribution.
Signal smoothed with EMA(3) to avoid whipsaws in mild downtrend.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.momentum.roc import rate_of_change
from indicators.volume.klinger import klinger
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV5(TrendingStrategy):
    """Donchian(20) lower break + ROC(10) < 0 + Klinger(12) < 0."""

    name = "short_I_1h_v5"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    dc_period: int = 20
    roc_period: int = 10
    klinger_fast: int = 12
    chandelier_mult: float = 2.6

    _dc_upper: np.ndarray | None = None
    _dc_lower: np.ndarray | None = None
    _dc_mid: np.ndarray | None = None
    _roc_arr: np.ndarray | None = None
    _klinger_line: np.ndarray | None = None
    _klinger_sig: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

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
        self._roc_arr = rate_of_change(self._closes, self.roc_period)
        self._klinger_line, self._klinger_sig = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=55, signal=13,
        )

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        dl = self._dc_lower[bar_index]
        dm = self._dc_mid[bar_index]
        roc = self._roc_arr[bar_index]
        kl = self._klinger_line[bar_index]

        if any(np.isnan(v) for v in (dl, dm, roc, kl)):
            return 0.0

        if c > dl:
            return 0.0

        depth = (dm - c) / dm if dm > 0 else 0.0
        signal = -(0.3 + min(0.15, depth * 5.0))

        if roc < 0:
            signal -= min(0.1, abs(roc) * 0.02)

        if kl < 0:
            signal -= 0.1

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"Donchian Upper({self.dc_period})", "array": self._dc_upper,
             "type": "overlay", "style": "dash", "color": "#66bb6a"},
            {"name": f"Donchian Lower({self.dc_period})", "array": self._dc_lower,
             "type": "overlay", "style": "dash", "color": "#ef5350"},
            {"name": f"ROC({self.roc_period})", "array": self._roc_arr, "type": "subplot", "zero_line": True},
            {"name": "Klinger", "array": self._klinger_line, "type": "subplot", "panel": "Klinger", "zero_line": True},
            {"name": "Klinger Sig", "array": self._klinger_sig, "type": "subplot",
             "panel": "Klinger", "style": "dash", "color": "#ffab40"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper,
                                   style="dash", color="#66bb6a"),
                self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower,
                                   style="dash", color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"ROC({self.roc_period})",
                    [self._make_subplot_trace("ROC", datetimes, self._roc_arr, color="#bb86fc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Klinger",
                    [
                        self._make_subplot_trace("KVO", datetimes, self._klinger_line, color="#42a5f5"),
                        self._make_subplot_trace("Signal", datetimes, self._klinger_sig,
                                                 style="dash", color="#ffab40"),
                    ],
                    zero_line=True,
                ),
            ],
        }
