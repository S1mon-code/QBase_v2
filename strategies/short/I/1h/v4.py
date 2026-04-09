"""MildTrendShortI1hV4 — HMA slope + CCI bearish + MFI weak.

Economic logic: HMA(16) declining slope on 1H provides low-lag trend direction.
CCI(12) < -25 confirms price is trading below its statistical mean. MFI(12) < 42
validates weak money flow from institutional sellers. Signal smoothed with EMA(3)
to reduce overtrading in mild downtrend regime.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.momentum.cci import cci
from indicators.volume.mfi import mfi
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV4(TrendingStrategy):
    """HMA(16) slope < 0 + CCI(12) < -25 + MFI(12) < 42."""

    name = "short_I_1h_v4"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    hma_period: int = 16
    cci_period: int = 12
    mfi_period: int = 12
    chandelier_mult: float = 3.0

    _hma_line: np.ndarray | None = None
    _cci_arr: np.ndarray | None = None
    _mfi_arr: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

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
        self._hma_line = hma(self._closes, self.hma_period)
        self._cci_arr = cci(self._highs, self._lows, self._closes, self.cci_period)
        self._mfi_arr = mfi(self._highs, self._lows, self._closes, self._volumes, self.mfi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        h = self._hma_line[bar_index]
        h_prev = self._hma_line[bar_index - 1]
        c = self._cci_arr[bar_index]
        m = self._mfi_arr[bar_index]

        if any(np.isnan(v) for v in (h, h_prev, c, m)):
            return 0.0

        slope = h - h_prev
        if slope >= 0:
            return 0.0

        signal = -0.3

        if c < -25:
            signal -= min(0.2, abs(c + 25) / 500.0)

        if m < 42:
            signal -= min(0.15, (42 - m) / 100.0)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma_line, "type": "overlay", "color": "#26c6da"},
            {"name": f"CCI({self.cci_period})", "array": self._cci_arr, "type": "subplot",
             "zero_line": True, "horizontal_lines": [-25]},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi_arr, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [20, 42, 80]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma_line, color="#26c6da"),
            ],
            "subplots": [
                self._make_subplot(
                    f"CCI({self.cci_period})",
                    [self._make_subplot_trace("CCI", datetimes, self._cci_arr, color="#bb86fc")],
                    zero_line=True, horizontal_lines=[-25],
                ),
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi_arr, color="#ffab40")],
                    horizontal_lines=[20, 42, 80], y_range=[0, 100],
                ),
            ],
        }
