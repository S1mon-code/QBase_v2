"""StrongTrendShortAG1hV15 — KAMA(25) slope neg + MFI(8) < 35.

Economic logic: KAMA(25) adapts its smoothing to 1H silver volatility. Negative
slope confirms sustained downward direction. MFI(8) below 35 validates money
flowing out of the asset on a short-term basis — sellers dominating intraday.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.kama import kama
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV15(TrendingStrategy):
    """KAMA(25) slope < 0 + MFI(8) < 35."""

    name = "short_AG_1h_v15"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    kama_period: int = 25
    mfi_period: int = 8
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama_line = kama(self._closes, self.kama_period)
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes, self.mfi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, i: int) -> float:
        k = self._kama_line[i]
        k_prev = self._kama_line[i - 1] if i > 0 else np.nan
        m = self._mfi[i]

        if any(np.isnan(v) for v in (k, k_prev, m)):
            return 0.0

        slope = k - k_prev
        if slope >= 0 or m >= 35.0:
            return 0.0

        strength = -0.45
        if m < 25.0:
            strength -= 0.25
        if slope < -0.5:
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
            {"name": f"MFI({self.mfi_period})", "array": self._mfi,
             "panel": "MFI", "y_range": [0, 100], "horizontal_lines": [20, 35, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(f"MFI({self.mfi_period})", [
                    self._make_subplot_trace("MFI", datetimes, self._mfi, color="#42a5f5"),
                ], horizontal_lines=[20, 35, 80], y_range=[0, 100]),
            ],
        }
