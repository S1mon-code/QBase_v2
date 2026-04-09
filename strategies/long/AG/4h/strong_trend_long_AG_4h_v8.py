"""StrongTrendLongAG4hV8 — Donchian(50) + PPO(18,45,14) + Klinger(60).

Economic logic: Donchian channel breakout on 4H captures AG's range expansions.
PPO normalizes momentum across Silver's varying price levels. Klinger Volume
Oscillator (substituted for SmartMoney) tracks institutional accumulation.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.momentum.ppo import ppo
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG4hV8(TrendingStrategy):
    name = "long_AG_4h_v8"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    donchian_period: int = 50
    ppo_fast: int = 18
    ppo_slow: int = 45
    klinger_fast: int = 34
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.donchian_period,
        )
        self._ppo_line, self._ppo_signal, _ = ppo(
            self._closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=14,
        )
        self._klinger_line, self._klinger_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=60, signal=13,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_mid = self._dc_mid[bar_index]
        pp = self._ppo_line[bar_index]
        ps = self._ppo_signal[bar_index]
        kl = self._klinger_line[bar_index]
        ks = self._klinger_signal[bar_index]

        if any(np.isnan(v) for v in (close, dc_mid, pp, ps, kl, ks)):
            return 0.0

        if close > dc_mid and pp > ps and kl > ks:
            strength = min(1.0, abs(pp - ps) / 1.5 * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"DC Upper({self.donchian_period})", "array": self._dc_upper, "type": "overlay"},
            {"name": f"DC Mid({self.donchian_period})", "array": self._dc_mid, "type": "overlay"},
            {"name": f"DC Lower({self.donchian_period})", "array": self._dc_lower, "type": "overlay"},
            {"name": f"PPO({self.ppo_fast},{self.ppo_slow})", "array": self._ppo_line,
             "type": "subplot", "zero_line": True},
            {"name": "Klinger", "array": self._klinger_line, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"DC Upper", datetimes, self._dc_upper, color="#66bb6a"),
                self._make_overlay(f"DC Mid", datetimes, self._dc_mid, style="dash", color="#ffab40"),
                self._make_overlay(f"DC Lower", datetimes, self._dc_lower, color="#ef5350"),
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
