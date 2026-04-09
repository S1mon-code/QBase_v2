"""StrongTrendShortAG2hV4 — HMA Slope + CCI Bearish + Klinger Below Signal.

Economic logic: HMA(30) slope < 0 provides low-lag trend direction. CCI(20) < -40
confirms price is trending below its statistical mean. Klinger(18) below its signal
line validates bearish volume flow. EMA(4) smoothing prevents overtrading.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.hma import hma
from indicators.momentum.cci import cci
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV4(TrendingStrategy):
    """HMA(30) slope < 0 + CCI(20) < -40 + Klinger(18) < signal.

    Signal logic (raw, pre-smoothing):
        All 3 conditions met -> -0.85
        HMA slope < 0 AND CCI < -40 -> -0.55
        HMA slope < 0 AND Klinger < signal -> -0.35
        else -> 0.0
    """

    name = "strong_trend_short_AG_2h_v4"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 65

    hma_period: int = 30
    cci_period: int = 20
    klinger_fast: int = 18
    chandelier_mult: float = 3.3

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._klinger_line, self._klinger_sig = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=55, signal=13,
        )

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        h = self._hma[i]
        h_prev = self._hma[i - 1] if i > 0 else np.nan
        c = self._cci[i]
        kl = self._klinger_line[i]
        ks = self._klinger_sig[i]

        if any(np.isnan(v) for v in (h, h_prev, c, kl, ks)):
            return 0.0

        hma_slope_neg = h < h_prev
        cci_bearish = c < -40.0
        klinger_bearish = kl < ks

        if hma_slope_neg and cci_bearish and klinger_bearish:
            return -0.85
        if hma_slope_neg and cci_bearish:
            return -0.55
        if hma_slope_neg and klinger_bearish:
            return -0.35
        return 0.0

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma,
             "type": "overlay"},
            {"name": f"CCI({self.cci_period})", "array": self._cci,
             "zero_line": True, "horizontal_lines": [-40]},
            {"name": "Klinger", "array": self._klinger_line,
             "panel": "Klinger", "zero_line": True},
            {"name": "Klinger Sig", "array": self._klinger_sig,
             "panel": "Klinger"},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"HMA({self.hma_period})", datetimes,
                               self._hma, color="#ffab40"),
        ]
        subplots = [
            self._make_subplot(f"CCI({self.cci_period})", [
                self._make_subplot_trace("CCI", datetimes, self._cci, color="#42a5f5"),
            ], zero_line=True, horizontal_lines=[-40]),
            self._make_subplot("Klinger", [
                self._make_subplot_trace("KVO", datetimes, self._klinger_line,
                                         color="#66bb6a"),
                self._make_subplot_trace("Signal", datetimes, self._klinger_sig,
                                         color="#ef5350"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
