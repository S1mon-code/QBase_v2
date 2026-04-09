"""MildTrendLongI2hV9 — TEMA(45) + Stochastic(40,10) + VolumeSpike(40).

Economic logic: TEMA's triple smoothing captures 2H iron ore trends with minimal lag.
Stochastic in the 30-85 zone confirms bullish momentum without overbought risk.
Volume spikes validate conviction behind trend moves. Signal scales with stochastic
level and is boosted during volume expansion.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.stochastic import stochastic
from indicators.trend.tema import tema
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV9(TrendingStrategy):
    name = "long_I_2h_v9"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 55

    tema_period: int = 45
    stoch_k: int = 40
    stoch_d: int = 10
    vs_period: int = 40
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, period=self.tema_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes, k_period=self.stoch_k, d_period=self.stoch_d
        )
        self._vol_spike = volume_spike(self._volumes, period=self.vs_period)

    def _generate_signal(self, bar_index: int) -> float:
        t = self._tema[bar_index]
        t_prev = self._tema[bar_index - 1] if bar_index > 0 else np.nan
        sk = self._stoch_k[bar_index]
        sd = self._stoch_d[bar_index]

        if any(np.isnan(v) for v in [t, t_prev, sk, sd]):
            return 0.0

        tema_rising = t > t_prev
        stoch_bullish = 30.0 < sk < 85.0 and sk > sd

        if not (tema_rising and stoch_bullish):
            return 0.0

        stoch_score = min(1.0, (sk - 30.0) / 50.0) * 0.4
        spike_bonus = 0.15 if self._vol_spike[bar_index] else 0.0
        return min(1.0, 0.3 + stoch_score + spike_bonus + 0.1)

    def get_indicator_config(self):
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay"},
            {"name": "Stoch %K", "array": self._stoch_k, "type": "subplot", "panel": "Stochastic",
             "y_range": [0, 100], "horizontal_lines": [20, 80]},
            {"name": "Stoch %D", "array": self._stoch_d, "type": "subplot", "panel": "Stochastic", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ffab40"),
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
            ],
        }
