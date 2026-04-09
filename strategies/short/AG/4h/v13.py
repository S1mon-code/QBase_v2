"""StrongTrendShortAG4hV13 — MACD(14,30,9) bearish + Force Index(14) < 0.

Economic logic: MACD(14,30,9) on 4H silver provides a balanced momentum reading
for the intermediate timeframe. MACD line below signal confirms bearish trend.
Force Index(14) negative validates that selling is volume-backed — institutional
distribution supports the downtrend.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.ema import ema
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV13(TrendingStrategy):
    """MACD(14,30,9) bearish + Force Index(14) < 0."""

    name = "short_AG_4h_v13"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 45

    macd_fast: int = 14
    macd_slow: int = 30
    macd_signal: int = 9
    fi_period: int = 14
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._macd_line, self._macd_sig, self._macd_hist = macd(
            self._closes, self.macd_fast, self.macd_slow, self.macd_signal,
        )
        self._fi = force_index(self._closes, self._volumes, self.fi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        ml = self._macd_line[i]
        ms = self._macd_sig[i]
        f = self._fi[i]

        if any(np.isnan(v) for v in (ml, ms, f)):
            return 0.0

        if ml >= ms or f >= 0:
            return 0.0

        strength = -0.50
        spread = ms - ml
        if spread > 0.5:
            strength -= 0.25
        if f < -1000:
            strength -= 0.15
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
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "panel": "FI", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot("MACD", [
                    self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#42a5f5"),
                    self._make_subplot_trace("Signal", datetimes, self._macd_sig, color="#ff9800"),
                ], zero_line=True),
                self._make_subplot(f"Force Index({self.fi_period})", [
                    self._make_subplot_trace("FI", datetimes, self._fi, color="#7e57c2"),
                ], zero_line=True),
            ],
        }
