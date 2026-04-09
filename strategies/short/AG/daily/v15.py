"""StrongTrendShortAGDailyV15 — KAMA(70) slope neg + RSI(18) < 42 + Force Index(18) < 0.

Economic logic: KAMA(70) negative slope adapts to silver's volatility while
confirming the long-term downtrend. RSI(18) below 42 signals persistent selling
pressure. Force Index negative confirms that the selling is backed by volume —
real institutional distribution rather than a thin-market dip.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.ema import ema
from indicators.trend.kama import kama
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV15(TrendingStrategy):
    """KAMA(70) slope < 0 + RSI(18) < 42 + Force Index(18) < 0."""

    name = "short_AG_daily_v15"
    horizon = "slow"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    kama_period: int = 70
    rsi_period: int = 18
    fi_period: int = 18
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama_line = kama(self._closes, self.kama_period)
        self._rsi = rsi(self._closes, self.rsi_period)
        self._fi = force_index(self._closes, self._volumes, self.fi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, i: int) -> float:
        k = self._kama_line[i]
        k_prev = self._kama_line[i - 1] if i > 0 else np.nan
        r = self._rsi[i]
        f = self._fi[i]

        if any(np.isnan(v) for v in (k, k_prev, r, f)):
            return 0.0

        slope = k - k_prev
        if slope >= 0 or r >= 42.0 or f >= 0:
            return 0.0

        strength = -0.40
        if r < 35.0:
            strength -= 0.25
        if slope < -1.0:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama_line, "type": "overlay"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi,
             "panel": "RSI", "y_range": [0, 100], "horizontal_lines": [30, 42, 70]},
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "panel": "FI", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(f"RSI({self.rsi_period})", [
                    self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc"),
                ], horizontal_lines=[30, 42, 70], y_range=[0, 100]),
                self._make_subplot(f"Force Index({self.fi_period})", [
                    self._make_subplot_trace("FI", datetimes, self._fi, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
