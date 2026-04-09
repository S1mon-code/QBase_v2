"""StrongTrendLongAG1hV6 — TEMA(22) + Vortex(22) + Twiggs(25).

Economic logic: TEMA triple-smoothing for fast 1H AG trend detection. Vortex
separates positive/negative trend movements for directional clarity. Twiggs
Money Flow confirms volume-driven accumulation during Silver's fast uptrends.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.tema import tema
from indicators.trend.vortex import vortex
from indicators.volume.twiggs import twiggs_money_flow
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG1hV6(TrendingStrategy):
    name = "long_AG_1h_v6"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    tema_period: int = 22
    vortex_period: int = 22
    twiggs_period: int = 25
    chandelier_mult: float = 2.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, period=self.tema_period)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period,
        )
        self._twiggs = twiggs_money_flow(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.twiggs_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        t = self._tema[bar_index]
        t_prev = self._tema[bar_index - 1]
        vip = self._vi_plus[bar_index]
        vim = self._vi_minus[bar_index]
        tw = self._twiggs[bar_index]

        if any(np.isnan(v) for v in (t, t_prev, vip, vim, tw)):
            return 0.0

        if t > t_prev and vip > vim and tw > 0.0:
            diff = vip - vim
            strength = min(1.0, diff * 2.5 + tw * 2.0 + 0.1)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay"},
            {"name": f"VI+({self.vortex_period})", "array": self._vi_plus,
             "type": "subplot", "panel": "Vortex"},
            {"name": f"VI-({self.vortex_period})", "array": self._vi_minus,
             "type": "subplot", "panel": "Vortex", "horizontal_lines": [1.0]},
            {"name": f"Twiggs({self.twiggs_period})", "array": self._twiggs,
             "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Vortex({self.vortex_period})",
                    [self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#66bb6a"),
                     self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350")],
                    horizontal_lines=[1.0],
                ),
                self._make_subplot(
                    f"Twiggs({self.twiggs_period})",
                    [self._make_subplot_trace("Twiggs", datetimes, self._twiggs, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
