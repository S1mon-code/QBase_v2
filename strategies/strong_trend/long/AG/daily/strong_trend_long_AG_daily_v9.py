"""StrongTrendLongAGDailyV9 — TEMA(100) + Aroon(120) + BSP(80).

Economic logic: TEMA triple exponential smoothing provides fast response to
Silver's daily trend shifts while filtering noise. Aroon with long period
confirms trend persistence. Buying/Selling Pressure decomposes volume into
directional components, verifying accumulation in AG.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.tema import tema
from indicators.trend.aroon import aroon
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAGDailyV9(TrendingStrategy):
    """TEMA trend + Aroon persistence + Buying/Selling Pressure.

    Signal logic:
        - TEMA rising AND Aroon oscillator > 0 AND buy_pressure > sell_pressure → long
        - Strength scales with Aroon oscillator magnitude

    Attributes:
        tema_period:   TEMA period.
        aroon_period:  Aroon lookback.
        bsp_period:    Buying/Selling Pressure smoothing.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "strong_trend_long_AG_daily_v9"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 200

    tema_period: int = 100
    aroon_period: int = 120
    bsp_period: int = 80
    chandelier_mult: float = 3.5

    def on_init_arrays(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        volumes: np.ndarray,
        oi: np.ndarray,
        datetimes: np.ndarray,
    ) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, period=self.tema_period)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period,
        )
        self._buy_p, self._sell_p = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.bsp_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        t = self._tema[bar_index]
        t_prev = self._tema[bar_index - 1]
        aroon_osc = self._aroon_osc[bar_index]
        bp = self._buy_p[bar_index]
        sp = self._sell_p[bar_index]

        if any(np.isnan(v) for v in (t, t_prev, aroon_osc, bp, sp)):
            return 0.0

        tema_rising = t > t_prev
        aroon_bull = aroon_osc > 0.0
        pressure_bull = bp > sp

        if tema_rising and aroon_bull and pressure_bull:
            strength = min(1.0, aroon_osc / 100.0 * 0.6 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay"},
            {"name": f"Aroon Osc({self.aroon_period})",
             "array": self._aroon_osc, "type": "subplot", "zero_line": True,
             "y_range": [-100, 100]},
            {"name": f"Buy Pressure({self.bsp_period})",
             "array": self._buy_p, "type": "subplot", "panel": "BSP"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Aroon({self.aroon_period})",
                    [self._make_subplot_trace("Aroon Up", datetimes, self._aroon_up, color="#66bb6a"),
                     self._make_subplot_trace("Aroon Down", datetimes, self._aroon_down, color="#ef5350")],
                    y_range=[0, 100],
                ),
                self._make_subplot(
                    f"BSP({self.bsp_period})",
                    [self._make_subplot_trace("Buy", datetimes, self._buy_p, color="#66bb6a"),
                     self._make_subplot_trace("Sell", datetimes, self._sell_p, color="#ef5350")],
                ),
            ],
        }
