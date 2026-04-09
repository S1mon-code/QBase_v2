"""MildTrendLongI4hV3 — KAMA(50) + Stochastic(30,10) + VolumeMomentum(40).

Economic logic: KAMA adapts speed to iron ore's 4H efficiency ratio — fast in trends,
slow in consolidation. Stochastic in the 50-80 zone confirms momentum without
overbought risk. Volume momentum ratio validates participation. Signal scales with
stochastic level and volume conviction.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.stochastic import stochastic
from indicators.trend.kama import kama
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV3(TrendingStrategy):
    name = "mild_trend_long_I_4h_v3"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 65

    kama_period: int = 50
    stoch_k: int = 30
    stoch_d: int = 10
    vol_mom_period: int = 40
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes, k_period=self.stoch_k, d_period=self.stoch_d
        )
        self._vol_mom = volume_momentum(self._volumes, period=self.vol_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1] if bar_index > 0 else np.nan
        sk = self._stoch_k[bar_index]
        vm = self._vol_mom[bar_index]
        close = self._closes[bar_index]

        if any(np.isnan(v) for v in [k, k_prev, sk, vm, close]):
            return 0.0

        kama_rising = k > k_prev
        stoch_bullish = 30.0 < sk < 85.0
        vol_strong = vm > 1.0

        if not (kama_rising and stoch_bullish and vol_strong):
            return 0.0

        stoch_score = min(1.0, (sk - 30.0) / 50.0) * 0.4
        vol_score = min(1.0, (vm - 1.0) / 0.5) * 0.3
        return min(1.0, 0.3 + stoch_score + vol_score)

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": "Stoch %K", "array": self._stoch_k, "type": "subplot", "panel": "Stochastic",
             "y_range": [0, 100], "horizontal_lines": [20, 80]},
            {"name": "Stoch %D", "array": self._stoch_d, "type": "subplot", "panel": "Stochastic", "style": "dash"},
            {"name": "Vol Mom", "array": self._vol_mom, "type": "subplot", "horizontal_lines": [1.0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Stochastic",
                    [
                        self._make_subplot_trace("%K", datetimes, self._stoch_k, color="#42a5f5"),
                        self._make_subplot_trace("%D", datetimes, self._stoch_d, color="#ff8a80", style="dash"),
                    ],
                    y_range=[0, 100], horizontal_lines=[20, 80],
                ),
                self._make_subplot(
                    "Volume Momentum",
                    [self._make_subplot_trace("VM", datetimes, self._vol_mom, color="#66bb6a")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
