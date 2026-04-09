"""StrongTrendLongAG2hV6 — FRAMA(60) + PPO(14,38,10) + Klinger(70).

Economic logic: FRAMA adapts to AG's 2H fractal structure. PPO normalizes
momentum across Silver's price range. Klinger Volume Oscillator (substituted
for SmartMoney) tracks institutional accumulation/distribution.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.frama import frama
from indicators.momentum.ppo import ppo
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG2hV6(TrendingStrategy):
    name = "long_AG_2h_v6"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 100

    frama_period: int = 60
    ppo_fast: int = 14
    ppo_slow: int = 38
    klinger_slow: int = 70
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._frama = frama(self._closes, period=self.frama_period)
        self._ppo_line, self._ppo_signal, _ = ppo(
            self._closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=10,
        )
        self._klinger_line, self._klinger_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=34, slow=self.klinger_slow, signal=13,
        )

    def _generate_signal(self, bar_index: int) -> float:
        f = self._frama[bar_index]
        f_prev = self._frama[bar_index - 1]
        pp = self._ppo_line[bar_index]
        ps = self._ppo_signal[bar_index]
        kl = self._klinger_line[bar_index]
        ks = self._klinger_signal[bar_index]

        if any(np.isnan(v) for v in (f, f_prev, pp, ps, kl, ks)):
            return 0.0

        if f > f_prev and pp > ps and kl > ks:
            strength = min(1.0, abs(pp - ps) / 1.5 * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"FRAMA({self.frama_period})", "array": self._frama, "type": "overlay"},
            {"name": f"PPO({self.ppo_fast},{self.ppo_slow})", "array": self._ppo_line,
             "type": "subplot", "zero_line": True},
            {"name": "Klinger", "array": self._klinger_line,
             "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"FRAMA({self.frama_period})", datetimes, self._frama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"PPO({self.ppo_fast},{self.ppo_slow})",
                    [self._make_subplot_trace("PPO", datetimes, self._ppo_line, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._ppo_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Klinger",
                    [self._make_subplot_trace("Klinger", datetimes, self._klinger_line, color="#42a5f5"),
                     self._make_subplot_trace("Signal", datetimes, self._klinger_signal, color="#ff8a65")],
                    zero_line=True,
                ),
            ],
        }
