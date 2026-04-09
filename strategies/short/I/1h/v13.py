"""MildTrendShortI1hV13 — MACD bearish + CMF distribution.

Economic logic: MACD(8,18,6) line below signal on 1H Iron Ore identifies fast
bearish momentum. CMF(12) < 0 confirms institutional money is flowing out,
adding volume-based conviction to the short signal.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.ema import ema
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV13(TrendingStrategy):
    """MACD(8,18,6) bearish + CMF(12) < 0."""

    name = "short_I_1h_v13"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    macd_fast: int = 8
    macd_slow: int = 18
    macd_signal: int = 6
    cmf_period: int = 12
    chandelier_mult: float = 2.8

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
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        ml = self._macd_line[bar_index]
        ms = self._macd_sig[bar_index]
        c = self._cmf[bar_index]

        if any(np.isnan(v) for v in (ml, ms, c)):
            return 0.0

        if ml >= ms:
            return 0.0

        macd_gap = abs(ml - ms)
        signal = -(0.3 + min(0.2, macd_gap * 1.0))

        if c < 0:
            signal -= min(0.15, abs(c) * 0.5)

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
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
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
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26c6da")],
                    zero_line=True,
                ),
            ],
        }
