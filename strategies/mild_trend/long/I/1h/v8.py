"""MildTrendLongI1hV8 — PSAR + PPO(10,25,8) + VolumeSpike(20).

Economic logic: Parabolic SAR provides adaptive trailing stops for 1H iron ore trends.
PPO above zero with positive histogram confirms momentum acceleration. Volume spikes
validate conviction behind trend moves. Signal scales with PPO strength and is
boosted during volume expansion.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.ppo import ppo
from indicators.trend.psar import psar
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV8(TrendingStrategy):
    name = "mild_trend_long_I_1h_v8"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 35

    ppo_fast: int = 10
    ppo_slow: int = 25
    vs_period: int = 20
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._psar_vals, self._psar_dir = psar(self._highs, self._lows)
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(
            self._closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=8
        )
        self._vol_spike = volume_spike(self._volumes, period=self.vs_period)

    def _generate_signal(self, bar_index: int) -> float:
        pd = self._psar_dir[bar_index]
        ppo_val = self._ppo_line[bar_index]
        ppo_h = self._ppo_hist[bar_index]

        if any(np.isnan(v) for v in [pd, ppo_val, ppo_h]):
            return 0.0

        psar_bullish = pd == 1.0
        ppo_bullish = ppo_val > 0.0 and ppo_h > 0.0

        if not (psar_bullish and ppo_bullish):
            return 0.0

        ppo_score = min(1.0, ppo_val / 1.5) * 0.4
        spike_bonus = 0.15 if self._vol_spike[bar_index] else 0.0
        return min(1.0, 0.35 + ppo_score + spike_bonus)

    def get_indicator_config(self):
        return [
            {"name": "PSAR", "array": self._psar_vals, "type": "overlay", "style": "step"},
            {"name": "PPO Line", "array": self._ppo_line, "type": "subplot", "panel": "PPO", "zero_line": True},
            {"name": "PPO Signal", "array": self._ppo_signal, "type": "subplot", "panel": "PPO", "style": "dash"},
            {"name": "PPO Hist", "array": self._ppo_hist, "type": "subplot", "panel": "PPO", "style": "bar",
             "color_positive": "#66bb6a", "color_negative": "#ef5350"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("PSAR", datetimes, self._psar_vals, style="step", color="#ff7043"),
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
            ],
        }
