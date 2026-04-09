"""StrongTrendShortAG4hV10 — Keltner Breakdown + PPO Negative + MFI Weak.

Economic logic: Price below Keltner lower (2.8x ATR) on 4H signals extreme bearish
for silver. PPO < 0 confirms percentage-based momentum is negative. MFI < 40
shows money flow is leaving the asset.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.ppo import ppo
from indicators.trend.keltner import keltner
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV10(TrendingStrategy):
    """Keltner breakdown + PPO negative + MFI weak.

    Signal logic:
        close < Kelt_lower AND PPO_line < 0 AND MFI < 35 -> -0.90
        close < Kelt_mid AND PPO_line < 0 -> -0.45
        else -> 0.0
    """

    name = "short_AG_4h_v10"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    kelt_period: int = 40
    kelt_mult: float = 2.8
    mfi_period: int = 35
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_period, atr_period=self.kelt_period,
            multiplier=self.kelt_mult,
        )
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(
            self._closes, fast=16, slow=42, signal=13,
        )
        self._mfi = mfi(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.mfi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        kl = self._kelt_lower[bar_index]
        km = self._kelt_mid[bar_index]
        p = self._ppo_line[bar_index]
        m = self._mfi[bar_index]

        if np.isnan(c) or np.isnan(kl) or np.isnan(p):
            return 0.0

        if c < kl and p < 0 and (not np.isnan(m) and m < 35):
            return -0.90
        if c < km and p < 0:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"Kelt Upper({self.kelt_period})", "array": self._kelt_upper, "type": "overlay", "style": "dash"},
            {"name": f"Kelt Mid({self.kelt_period})", "array": self._kelt_mid, "type": "overlay"},
            {"name": f"Kelt Lower({self.kelt_period})", "array": self._kelt_lower, "type": "overlay", "style": "dash"},
            {"name": "PPO Line", "array": self._ppo_line, "panel": "PPO"},
            {"name": "PPO Signal", "array": self._ppo_signal, "panel": "PPO"},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi,
             "y_range": [0, 100], "horizontal_lines": [20, 35, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"Kelt Upper({self.kelt_period})", datetimes, self._kelt_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"Kelt Mid({self.kelt_period})", datetimes, self._kelt_mid, color="#ffab40"),
            self._make_overlay(f"Kelt Lower({self.kelt_period})", datetimes, self._kelt_lower, style="dash", color="#26a69a"),
        ]
        subplots = [
            self._make_subplot("PPO", [
                self._make_subplot_trace("PPO Line", datetimes, self._ppo_line, color="#42a5f5"),
                self._make_subplot_trace("PPO Signal", datetimes, self._ppo_signal, color="#ff7043"),
            ], zero_line=True),
            self._make_subplot(f"MFI({self.mfi_period})", [
                self._make_subplot_trace("MFI", datetimes, self._mfi, color="#ab47bc"),
            ], horizontal_lines=[20, 35, 80], y_range=[0, 100]),
        ]
        return {"overlays": overlays, "subplots": subplots}
