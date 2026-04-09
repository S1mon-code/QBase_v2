"""StrongTrendShortAG1hV4 — HMA(20) declining + CCI(14) < -30 + MFI(14) < 40.

Economic logic: HMA responds faster than standard MA with less lag, perfect for
catching silver's sharp 1H selloffs.  HMA declining (negative slope) is a
slope-based filter that avoids the cross-based whipsaw problem.  CCI < -30
confirms bearish price deviation from statistical mean.  MFI < 40 validates
weak money flow.  3-bar EMA smoothing on composite signal.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.trend.ema import ema
from indicators.trend.hma import hma
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV4(TrendingStrategy):
    """HMA(20) declining + CCI(14) < -30 + MFI(14) < 40."""

    name = "short_AG_1h_v4"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    hma_period: int = 20
    cci_period: int = 14
    mfi_period: int = 14
    smooth_period: int = 3
    chandelier_mult: float = 2.8

    _hma_line: np.ndarray | None = None
    _cci_line: np.ndarray | None = None
    _mfi_line: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._hma_line = hma(self._closes, self.hma_period)
        self._cci_line = cci(self._highs, self._lows, self._closes, self.cci_period)
        self._mfi_line = mfi(self._highs, self._lows, self._closes, self._volumes, self.mfi_period)

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=self.smooth_period)

    def _raw_signal(self, i: int) -> float:
        h = self._hma_line
        c = self._cci_line
        m = self._mfi_line

        if i < 1 or np.isnan(h[i]) or np.isnan(h[i - 1]) or np.isnan(c[i]) or np.isnan(m[i]):
            return 0.0

        # HMA declining (slope-based, no cross)
        slope = h[i] - h[i - 1]
        if slope >= 0.0:
            return 0.0

        # CCI confirms bearish deviation
        if c[i] >= -30.0:
            return 0.0

        # MFI shows weak money flow
        if m[i] >= 40.0:
            return 0.0

        strength = -0.55
        if c[i] < -100.0:
            strength -= 0.20
        if m[i] < 25.0:
            strength -= 0.25
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma_line, "type": "overlay"},
            {"name": "CCI(14)", "array": self._cci_line, "type": "subplot",
             "zero_line": True, "horizontal_lines": [-30, -100]},
            {"name": "MFI(14)", "array": self._mfi_line, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [25, 40]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot("CCI(14)", [
                    self._make_subplot_trace("CCI", datetimes, self._cci_line, color="#bb86fc"),
                ], zero_line=True, horizontal_lines=[-30, -100]),
                self._make_subplot("MFI(14)", [
                    self._make_subplot_trace("MFI", datetimes, self._mfi_line, color="#42a5f5"),
                ], horizontal_lines=[25, 40], y_range=[0, 100]),
            ],
        }
