"""StrongTrendLongAG4hV7 — TEMA(35) + Aroon(40) + VolumeMomentum(30).

Economic logic: TEMA triple-smoothing provides fast AG 4H trend detection.
Aroon confirms whether Silver is making new highs consistently. Volume
Momentum ratio tracks whether volume is expanding with trend.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.tema import tema
from indicators.trend.aroon import aroon
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG4hV7(TrendingStrategy):
    name = "strong_trend_long_AG_4h_v7"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    tema_period: int = 35
    aroon_period: int = 40
    volmom_period: int = 30
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, period=self.tema_period)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period,
        )
        self._volmom = volume_momentum(self._volumes, period=self.volmom_period)

    def _generate_signal(self, bar_index: int) -> float:
        t = self._tema[bar_index]
        t_prev = self._tema[bar_index - 1]
        ao = self._aroon_osc[bar_index]
        vm = self._volmom[bar_index]

        if any(np.isnan(v) for v in (t, t_prev, ao, vm)):
            return 0.0

        if t > t_prev and ao > 0.0 and vm > 1.0:
            strength = min(1.0, ao / 80.0 * 0.5 + min(vm - 1.0, 1.0) * 0.4 + 0.1)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay"},
            {"name": f"Aroon Osc({self.aroon_period})", "array": self._aroon_osc,
             "type": "subplot", "zero_line": True, "y_range": [-100, 100]},
            {"name": f"VolMom({self.volmom_period})", "array": self._volmom,
             "type": "subplot", "horizontal_lines": [1.0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Aroon({self.aroon_period})",
                    [self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#66bb6a"),
                     self._make_subplot_trace("Down", datetimes, self._aroon_down, color="#ef5350")],
                    y_range=[0, 100],
                ),
                self._make_subplot(
                    f"VolMom({self.volmom_period})",
                    [self._make_subplot_trace("VolMom", datetimes, self._volmom, color="#42a5f5")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
