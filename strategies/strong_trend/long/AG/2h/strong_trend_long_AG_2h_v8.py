"""StrongTrendLongAG2hV8 — TEMA(40) + Stochastic(35,10,10) + BSP(30).

Economic logic: TEMA triple-smoothing captures AG 2H trend direction. Slow
stochastic with wide periods catches momentum turns in Silver's volatile
sessions. Buying/Selling Pressure decomposes volume into directional components.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.tema import tema
from indicators.momentum.stochastic import stochastic
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG2hV8(TrendingStrategy):
    name = "strong_trend_long_AG_2h_v8"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    tema_period: int = 40
    stoch_k: int = 35
    stoch_d: int = 10
    bsp_period: int = 30
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, period=self.tema_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes,
            k_period=self.stoch_k, d_period=self.stoch_d,
        )
        self._buy_p, self._sell_p = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.bsp_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        t = self._tema[bar_index]
        t_prev = self._tema[bar_index - 1]
        sk = self._stoch_k[bar_index]
        bp = self._buy_p[bar_index]
        sp = self._sell_p[bar_index]

        if any(np.isnan(v) for v in (t, t_prev, sk, bp, sp)):
            return 0.0

        if t > t_prev and sk > 50.0 and bp > sp:
            strength = min(1.0, (sk - 50.0) / 30.0 * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay"},
            {"name": f"Stoch %K({self.stoch_k})", "array": self._stoch_k,
             "type": "subplot", "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
            {"name": f"BuyPressure({self.bsp_period})", "array": self._buy_p,
             "type": "subplot", "panel": "BSP"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Stochastic({self.stoch_k})",
                    [self._make_subplot_trace("%K", datetimes, self._stoch_k, color="#ab47bc"),
                     self._make_subplot_trace("%D", datetimes, self._stoch_d, color="#ff8a65")],
                    y_range=[0, 100], horizontal_lines=[20, 50, 80],
                ),
                self._make_subplot(
                    f"BSP({self.bsp_period})",
                    [self._make_subplot_trace("Buy", datetimes, self._buy_p, color="#66bb6a"),
                     self._make_subplot_trace("Sell", datetimes, self._sell_p, color="#ef5350")],
                ),
            ],
        }
