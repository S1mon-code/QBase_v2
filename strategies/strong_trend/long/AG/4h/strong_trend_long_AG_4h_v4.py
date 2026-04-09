"""StrongTrendLongAG4hV4 — KAMA(50) + Stochastic(30,10,10) + ForceIndex(30).

Economic logic: KAMA adapts to AG's 4H efficiency ratio, widening during
choppy sessions and tightening during directional moves. Slow stochastic
with wide periods captures momentum turns. Force Index confirms volume-
backed price movement.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.momentum.stochastic import stochastic
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG4hV4(TrendingStrategy):
    name = "strong_trend_long_AG_4h_v4"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    kama_period: int = 50
    stoch_k: int = 30
    stoch_d: int = 10
    force_period: int = 30
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes,
            k_period=self.stoch_k, d_period=self.stoch_d,
        )
        self._force = force_index(self._closes, self._volumes, period=self.force_period)

    def _generate_signal(self, bar_index: int) -> float:
        k_val = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1]
        sk = self._stoch_k[bar_index]
        fi = self._force[bar_index]

        if any(np.isnan(v) for v in (k_val, k_prev, sk, fi)):
            return 0.0

        if k_val > k_prev and sk > 50.0 and fi > 0.0:
            strength = min(1.0, (sk - 50.0) / 30.0 * 0.6 + 0.3)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": f"Stoch %K({self.stoch_k})", "array": self._stoch_k,
             "type": "subplot", "y_range": [0, 100], "horizontal_lines": [20, 50, 80],
             "panel": "Stochastic"},
            {"name": f"ForceIndex({self.force_period})", "array": self._force,
             "type": "subplot", "zero_line": True},
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
                    f"ForceIndex({self.force_period})",
                    [self._make_subplot_trace("Force", datetimes, self._force, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
