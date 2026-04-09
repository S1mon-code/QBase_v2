"""MildTrendLongI4hV9 — Keltner(40,2.0) + PPO(20,50,15) + BSP(30).

Economic logic: Keltner Channel provides ATR-based trend bands for 4H iron ore.
PPO (percentage MACD) above zero with positive histogram confirms acceleration.
Buying/selling pressure ratio validates volume-side conviction. Signal scales with
PPO strength and channel position.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.ppo import ppo
from indicators.trend.keltner import keltner
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV9(TrendingStrategy):
    name = "mild_trend_long_I_4h_v9"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 60

    kelt_period: int = 40
    kelt_mult: float = 2.0
    ppo_fast: int = 20
    ppo_slow: int = 50
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_period, atr_period=self.kelt_period // 2, multiplier=self.kelt_mult
        )
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(
            self._closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=15
        )
        self._buy_p, self._sell_p, self._pressure_ratio = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes, period=30
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        km = self._kelt_mid[bar_index]
        ku = self._kelt_upper[bar_index]
        ppo_val = self._ppo_line[bar_index]
        ppo_h = self._ppo_hist[bar_index]
        pr = self._pressure_ratio[bar_index]

        if any(np.isnan(v) for v in [close, km, ku, ppo_val, ppo_h, pr]):
            return 0.0

        above_mid = close > km
        ppo_bullish = ppo_val > 0.0 and ppo_h > 0.0
        buying_dominant = pr > 1.0

        if not (above_mid and ppo_bullish and buying_dominant):
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
            {"name": "Pressure Ratio", "array": self._pressure_ratio, "type": "subplot", "horizontal_lines": [1.0]},
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
                    "Buy/Sell Pressure",
                    [self._make_subplot_trace("Ratio", datetimes, self._pressure_ratio, color="#ab47bc")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
