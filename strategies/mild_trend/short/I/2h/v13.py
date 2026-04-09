"""MildTrendShortI2hV13 — MACD bearish + Force Index negative.

Economic logic: MACD(12,28,8) line below signal on 2H Iron Ore identifies
medium-term bearish momentum. Force Index(14) < 0 confirms volume-weighted
selling pressure, ensuring the bearish momentum has conviction.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.ema import ema
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV13(TrendingStrategy):
    """MACD(12,28,8) bearish + Force Index(14) < 0."""

    name = "mild_trend_short_I_2h_v13"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 45

    macd_fast: int = 12
    macd_slow: int = 28
    macd_signal: int = 8
    fi_period: int = 14
    chandelier_mult: float = 3.0

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
        self._macd_line, self._macd_sig, self._macd_hist = macd(
            self._closes, self.macd_fast, self.macd_slow, self.macd_signal,
        )
        self._fi = force_index(self._closes, self._volumes, self.fi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, bar_index: int) -> float:
        ml = self._macd_line[bar_index]
        ms = self._macd_sig[bar_index]
        fi = self._fi[bar_index]

        if any(np.isnan(v) for v in (ml, ms, fi)):
            return 0.0

        if ml >= ms:
            return 0.0

        macd_gap = abs(ml - ms)
        signal = -(0.3 + min(0.2, macd_gap * 0.8))

        if fi < 0:
            signal -= min(0.15, abs(fi) / 1e6 * 0.1)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "MACD Line", "array": self._macd_line, "type": "subplot", "color": "#2196f3"},
            {"name": "MACD Signal", "array": self._macd_sig, "type": "subplot",
             "panel": "MACD Line", "color": "#ff9800"},
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    "MACD",
                    [
                        self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#2196f3"),
                        self._make_subplot_trace("Signal", datetimes, self._macd_sig, color="#ff9800"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"Force Index({self.fi_period})",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#ef5350")],
                    zero_line=True,
                ),
            ],
        }
