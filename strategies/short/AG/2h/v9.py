"""StrongTrendShortAG2hV9 — Vortex Bearish + MACD Line Negative + EMV Negative.

Economic logic: Vortex(22) VI- > VI+ * 1.05 confirms strong bearish directional
movement. MACD line < 0 validates sustained negative momentum (not histogram to
avoid noise in stable trends). EMV(18) < 0 shows price moving down on low
volume effort — path of least resistance is down. EMA(4) smoothing.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.vortex import vortex
from indicators.momentum.macd import macd
from indicators.volume.emv import emv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV9(TrendingStrategy):
    """Vortex(22) VI- > VI+ * 1.05 + MACD line < 0 + EMV(18) < 0.

    Signal logic (raw, pre-smoothing):
        All 3 conditions met -> -0.90
        Vortex bearish AND MACD line < 0 -> -0.55
        Vortex bearish AND EMV < 0 -> -0.40
        else -> 0.0
    """

    name = "short_AG_2h_v9"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    vortex_period: int = 22
    emv_period: int = 18
    chandelier_mult: float = 3.4

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period,
        )
        self._macd_line, self._macd_sig, self._macd_hist = macd(
            self._closes, fast=12, slow=26, signal=9,
        )
        self._emv = emv(self._highs, self._lows, self._volumes,
                        period=self.emv_period)

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        vip = self._vi_plus[i]
        vim = self._vi_minus[i]
        ml = self._macd_line[i]
        ev = self._emv[i]

        if any(np.isnan(v) for v in (vip, vim, ml, ev)):
            return 0.0

        vortex_bearish = vim > vip * 1.05
        macd_neg = ml < 0.0
        emv_neg = ev < 0.0

        if vortex_bearish and macd_neg and emv_neg:
            return -0.90
        if vortex_bearish and macd_neg:
            return -0.55
        if vortex_bearish and emv_neg:
            return -0.40
        return 0.0

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "VI+", "array": self._vi_plus,
             "panel": "Vortex", "horizontal_lines": [1.0]},
            {"name": "VI-", "array": self._vi_minus, "panel": "Vortex"},
            {"name": "MACD Line", "array": self._macd_line,
             "panel": "MACD", "zero_line": True},
            {"name": "MACD Signal", "array": self._macd_sig, "panel": "MACD"},
            {"name": f"EMV({self.emv_period})", "array": self._emv, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        subplots = [
            self._make_subplot("Vortex", [
                self._make_subplot_trace("VI+", datetimes, self._vi_plus,
                                         color="#66bb6a"),
                self._make_subplot_trace("VI-", datetimes, self._vi_minus,
                                         color="#ef5350"),
            ], horizontal_lines=[1.0]),
            self._make_subplot("MACD", [
                self._make_subplot_trace("MACD Line", datetimes, self._macd_line,
                                         color="#42a5f5"),
                self._make_subplot_trace("Signal", datetimes, self._macd_sig,
                                         color="#ffab40"),
            ], zero_line=True),
            self._make_subplot(f"EMV({self.emv_period})", [
                self._make_subplot_trace("EMV", datetimes, self._emv, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": [], "subplots": subplots}
