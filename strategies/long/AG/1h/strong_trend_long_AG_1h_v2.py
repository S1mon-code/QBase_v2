"""StrongTrendLongAG1hV2 — HMA(25) + MACD(8,22,7) + BSP(15).

Economic logic: HMA with fast period captures Silver's explosive 1H moves
with minimal lag. Fast MACD parameters catch momentum surges early. BSP
confirms directional volume dominance during breakouts.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.momentum.macd import macd
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG1hV2(TrendingStrategy):
    name = "long_AG_1h_v2"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    hma_period: int = 25
    macd_fast: int = 8
    macd_slow: int = 22
    bsp_period: int = 15
    chandelier_mult: float = 2.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._macd_line, self._macd_signal, _ = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=7,
        )
        self._buy_p, self._sell_p = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.bsp_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1]
        ml = self._macd_line[bar_index]
        bp = self._buy_p[bar_index]
        sp = self._sell_p[bar_index]

        if any(np.isnan(v) for v in (h, h_prev, ml, bp, sp)):
            return 0.0

        if h > h_prev and ml > 0.0 and bp > sp:
            strength = min(1.0, abs(ml) / (abs(self._closes[bar_index]) * 0.005 + 1e-9) + 0.3)
            return max(0.0, min(1.0, strength))
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": f"MACD({self.macd_fast},{self.macd_slow})", "array": self._macd_line,
             "type": "subplot", "zero_line": True},
            {"name": f"BuyPressure({self.bsp_period})", "array": self._buy_p,
             "type": "subplot", "panel": "BSP"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"MACD({self.macd_fast},{self.macd_slow})",
                    [self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._macd_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"BSP({self.bsp_period})",
                    [self._make_subplot_trace("Buy", datetimes, self._buy_p, color="#66bb6a"),
                     self._make_subplot_trace("Sell", datetimes, self._sell_p, color="#ef5350")],
                ),
            ],
        }
