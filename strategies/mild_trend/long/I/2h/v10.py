"""MildTrendLongI2hV10 — Keltner(60,2.0) + PPO(15,40,10) + OI_Momentum(80).

Economic logic: Keltner Channel provides ATR-based trend bands for 2H iron ore.
PPO above zero with positive histogram confirms momentum acceleration. OI momentum
validates speculative commitment growing with the trend. Signal scales with PPO
strength and channel position.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.ppo import ppo
from indicators.trend.keltner import keltner
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV10(TrendingStrategy):
    name = "mild_trend_long_I_2h_v10"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 90

    kelt_period: int = 60
    kelt_mult: float = 2.0
    ppo_fast: int = 15
    ppo_slow: int = 40
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_period, atr_period=self.kelt_period // 2, multiplier=self.kelt_mult
        )
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(
            self._closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=10
        )
        self._oi_mom = oi_momentum(self._oi, period=80)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        km = self._kelt_mid[bar_index]
        ku = self._kelt_upper[bar_index]
        ppo_val = self._ppo_line[bar_index]
        ppo_h = self._ppo_hist[bar_index]
        oi_val = self._oi_mom[bar_index]

        if any(np.isnan(v) for v in [close, km, ku, ppo_val, ppo_h, oi_val]):
            return 0.0

        above_mid = close > km
        ppo_bullish = ppo_val > 0.0 and ppo_h > 0.0
        oi_expanding = oi_val > 0.0

        if not (above_mid and ppo_bullish and oi_expanding):
            return 0.0

        ppo_score = min(1.0, ppo_val / 2.0) * 0.4
        breakout = 0.2 if close > ku else 0.0
        return min(1.0, 0.3 + ppo_score + breakout)

    def get_indicator_config(self):
        return [
            {"name": "Keltner Upper", "array": self._kelt_upper, "type": "overlay", "style": "dash"},
            {"name": "Keltner Mid", "array": self._kelt_mid, "type": "overlay"},
            {"name": "Keltner Lower", "array": self._kelt_lower, "type": "overlay", "style": "dash"},
            {"name": "PPO Line", "array": self._ppo_line, "type": "subplot", "panel": "PPO", "zero_line": True},
            {"name": "PPO Signal", "array": self._ppo_signal, "type": "subplot", "panel": "PPO", "style": "dash"},
            {"name": "OI Mom", "array": self._oi_mom, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("Keltner Upper", datetimes, self._kelt_upper, style="dash", color="#78909c"),
                self._make_overlay("Keltner Mid", datetimes, self._kelt_mid, color="#ffab40"),
                self._make_overlay("Keltner Lower", datetimes, self._kelt_lower, style="dash", color="#78909c"),
            ],
            "subplots": [
                self._make_subplot(
                    "PPO",
                    [
                        self._make_subplot_trace("Line", datetimes, self._ppo_line, color="#42a5f5"),
                        self._make_subplot_trace("Signal", datetimes, self._ppo_signal, color="#ff8a80", style="dash"),
                        self._make_subplot_trace("Hist", datetimes, self._ppo_hist, style="bar",
                                                 color_positive="#66bb6a", color_negative="#ef5350"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "OI Momentum",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#4fc3f7")],
                    zero_line=True,
                ),
            ],
        }
