"""StrongTrendLongAG1hV5 — KAMA(30) + Stochastic(18,5,5) + VolumeMomentum(20).

Economic logic: KAMA adapts to AG's 1H efficiency for responsive trend capture.
Fast stochastic detects momentum surges in Silver's explosive sessions. Volume
Momentum tracks whether volume is expanding with trend.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.momentum.stochastic import stochastic
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG1hV5(TrendingStrategy):
    name = "long_AG_1h_v5"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    kama_period: int = 30
    stoch_k: int = 18
    stoch_d: int = 5
    volmom_period: int = 20
    chandelier_mult: float = 2.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes,
            k_period=self.stoch_k, d_period=self.stoch_d,
        )
        self._volmom = volume_momentum(self._volumes, period=self.volmom_period)

    def _generate_signal(self, bar_index: int) -> float:
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1]
        sk = self._stoch_k[bar_index]
        vm = self._volmom[bar_index]

        if any(np.isnan(v) for v in (k, k_prev, sk, vm)):
            return 0.0

        if k > k_prev and sk > 50.0 and vm > 1.0:
            strength = min(1.0, (sk - 50.0) / 30.0 * 0.4 + min(vm - 1.0, 1.0) * 0.4 + 0.2)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": f"Stoch %K({self.stoch_k})", "array": self._stoch_k,
             "type": "subplot", "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
            {"name": f"VolMom({self.volmom_period})", "array": self._volmom,
             "type": "subplot", "horizontal_lines": [1.0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Stochastic({self.stoch_k})",
                    [self._make_subplot_trace("%K", datetimes, self._stoch_k, color="#ab47bc"),
                     self._make_subplot_trace("%D", datetimes, self._stoch_d, color="#ff8a65")],
                    y_range=[0, 100], horizontal_lines=[20, 50, 80],
                ),
                self._make_subplot(
                    f"VolMom({self.volmom_period})",
                    [self._make_subplot_trace("VolMom", datetimes, self._volmom, color="#42a5f5")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
