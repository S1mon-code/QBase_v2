"""MildTrendShortI1hV15 — KAMA slope negative + MFI weak.

Economic logic: KAMA(30) declining on 1H Iron Ore adapts its smoothing to
market noise, so a negative slope is a reliable bearish signal. MFI(10) < 40
confirms money flow is net negative, with volume-weighted selling outpacing
buying pressure.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.trend.ema import ema
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV15(TrendingStrategy):
    """KAMA(30) slope < 0 + MFI(10) < 40."""

    name = "mild_trend_short_I_1h_v15"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 40

    kama_period: int = 30
    mfi_period: int = 10
    mfi_threshold: float = 40.0
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
        self._kama = kama(self._closes, self.kama_period)
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes, self.mfi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1]
        m = self._mfi[bar_index]

        if any(np.isnan(v) for v in (k, k_prev, m)):
            return 0.0

        slope = k - k_prev
        if slope >= 0:
            return 0.0

        slope_pct = abs(slope) / k_prev if k_prev != 0 else 0.0
        signal = -(0.3 + min(0.2, slope_pct * 500))

        if m < self.mfi_threshold:
            signal -= min(0.15, (self.mfi_threshold - m) / 100.0)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay", "color": "#ffab40"},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [20, 40, 80]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#ab47bc")],
                    horizontal_lines=[20, 40, 80], y_range=[0, 100],
                ),
            ],
        }
