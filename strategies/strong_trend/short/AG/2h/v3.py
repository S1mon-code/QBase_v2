"""StrongTrendShortAG2hV3 — Donchian Lower Break + ROC Negative + MFI Weak.

Economic logic: Price breaking below Donchian(35) lower channel signals new
low in the lookback window — strong trend continuation. ROC(15) < 0 confirms
negative price momentum. MFI(18) < 45 shows weak money inflow. EMA(4) signal
smoothing prevents overtrading on Donchian channel noise.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.donchian import donchian
from indicators.momentum.roc import rate_of_change
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV3(TrendingStrategy):
    """Donchian(35) lower break + ROC(15) < 0 + MFI(18) < 45.

    Signal logic (raw, pre-smoothing):
        All 3 conditions met -> -0.90
        Close < Donchian lower AND ROC < 0 -> -0.55
        Close < Donchian lower AND MFI < 45 -> -0.40
        else -> 0.0
    """

    name = "strong_trend_short_AG_2h_v3"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    dc_period: int = 35
    roc_period: int = 15
    mfi_period: int = 18
    chandelier_mult: float = 3.4

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.dc_period,
        )
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes,
                        period=self.mfi_period)

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        c = self._closes[i]
        dl = self._dc_lower[i]
        r = self._roc[i]
        m = self._mfi[i]

        if any(np.isnan(v) for v in (c, dl, r, m)):
            return 0.0

        below_dc = c < dl
        roc_neg = r < 0.0
        mfi_weak = m < 45.0

        if below_dc and roc_neg and mfi_weak:
            return -0.90
        if below_dc and roc_neg:
            return -0.55
        if below_dc and mfi_weak:
            return -0.40
        return 0.0

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"Donchian Upper({self.dc_period})", "array": self._dc_upper,
             "type": "overlay", "style": "dash"},
            {"name": f"Donchian Lower({self.dc_period})", "array": self._dc_lower,
             "type": "overlay", "style": "dash"},
            {"name": f"Donchian Mid({self.dc_period})", "array": self._dc_mid,
             "type": "overlay"},
            {"name": f"ROC({self.roc_period})", "array": self._roc, "zero_line": True},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi,
             "y_range": [0, 100], "horizontal_lines": [20, 45, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"DC Upper({self.dc_period})", datetimes,
                               self._dc_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"DC Lower({self.dc_period})", datetimes,
                               self._dc_lower, style="dash", color="#26a69a"),
            self._make_overlay(f"DC Mid({self.dc_period})", datetimes,
                               self._dc_mid, color="#ffab40"),
        ]
        subplots = [
            self._make_subplot(f"ROC({self.roc_period})", [
                self._make_subplot_trace("ROC", datetimes, self._roc, color="#42a5f5"),
            ], zero_line=True),
            self._make_subplot(f"MFI({self.mfi_period})", [
                self._make_subplot_trace("MFI", datetimes, self._mfi, color="#ab47bc"),
            ], horizontal_lines=[20, 45, 80], y_range=[0, 100]),
        ]
        return {"overlays": overlays, "subplots": subplots}
