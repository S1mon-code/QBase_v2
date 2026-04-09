"""StrongTrendShortAG2hV14 — Bollinger(20,2.0) below middle + Force Index(12) < 0.

Economic logic: Price below the Bollinger(20) middle band on 2H silver confirms
bearish bias. Force Index(12) negative validates that selling is volume-backed —
the downward move has institutional participation, not just retail noise.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV14(TrendingStrategy):
    """Bollinger(20,2.0) price < middle + Force Index(12) < 0."""

    name = "strong_trend_short_AG_2h_v14"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    bb_period: int = 20
    bb_std: float = 2.0
    fi_period: int = 12
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._bb_upper, self._bb_middle, self._bb_lower = bollinger_bands(
            self._closes, period=self.bb_period, num_std=self.bb_std,
        )
        self._fi = force_index(self._closes, self._volumes, self.fi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        c = self._closes[i]
        mid = self._bb_middle[i]
        lower = self._bb_lower[i]
        f = self._fi[i]

        if any(np.isnan(v) for v in (c, mid, lower, f)):
            return 0.0

        if c >= mid or f >= 0:
            return 0.0

        bb_range = mid - lower
        if bb_range <= 0:
            return 0.0
        depth = min(1.0, (mid - c) / bb_range)

        strength = -(0.35 + depth * 0.35)
        if f < -500:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "BB Upper", "array": self._bb_upper, "type": "overlay"},
            {"name": "BB Middle", "array": self._bb_middle, "type": "overlay"},
            {"name": "BB Lower", "array": self._bb_lower, "type": "overlay"},
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "panel": "FI", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("BB Upper", datetimes, self._bb_upper, color="#ef5350"),
                self._make_overlay("BB Middle", datetimes, self._bb_middle, color="#ffab40"),
                self._make_overlay("BB Lower", datetimes, self._bb_lower, color="#66bb6a"),
            ],
            "subplots": [
                self._make_subplot(f"Force Index({self.fi_period})", [
                    self._make_subplot_trace("FI", datetimes, self._fi, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
