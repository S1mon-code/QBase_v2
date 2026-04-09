"""MildTrendShortI1hV19 — TRIX bearish + OI momentum negative.

Economic logic: TRIX(10) < 0 on 1H Iron Ore, as a triple-smoothed momentum
indicator, signals persistent bearish pressure. OI momentum(12) < 0 shows
open interest is declining, indicating traders are unwinding longs and
adding to the bearish signal.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.trix import trix
from indicators.trend.ema import ema
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV19(TrendingStrategy):
    """TRIX(10) < 0 + OI momentum(12) < 0."""

    name = "short_I_1h_v19"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 40

    trix_period: int = 10
    oi_mom_period: int = 12
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
        self._trix, self._trix_signal = trix(self._closes, self.trix_period)
        self._oi_mom = oi_momentum(oi, self.oi_mom_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        t = self._trix[bar_index]
        om = self._oi_mom[bar_index]

        if any(np.isnan(v) for v in (t, om)):
            return 0.0

        if t >= 0:
            return 0.0

        signal = -(0.3 + min(0.2, abs(t) * 80))

        if om < 0:
            signal -= min(0.15, abs(om) * 0.3)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TRIX({self.trix_period})", "array": self._trix, "type": "subplot", "zero_line": True},
            {"name": f"OI Mom({self.oi_mom_period})", "array": self._oi_mom, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"TRIX({self.trix_period})",
                    [self._make_subplot_trace("TRIX", datetimes, self._trix, color="#ef5350")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"OI Momentum({self.oi_mom_period})",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#26c6da")],
                    zero_line=True,
                ),
            ],
        }
