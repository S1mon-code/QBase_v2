"""MildTrendShortI4hV4 — KAMA Below Price + Stochastic Bearish + OI Momentum Contraction.

Economic logic: Price below KAMA on 4H shows adaptive trend is bearish.
Stochastic %K below %D and both below 50 confirm bearish momentum alignment.
Negative OI momentum signals open interest is contracting, fueling the decline.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.momentum.stochastic import stochastic
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV4(TrendingStrategy):
    """Price below KAMA(40) + Stoch(25,8)<50 + OI_Momentum(35)<0."""

    name = "short_I_4h_v4"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 55

    kama_period: int = 40
    stoch_k: int = 25
    stoch_d: int = 8
    oi_mom_period: int = 35
    chandelier_mult: float = 2.5

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
        self._kama = kama(self._closes, period=self.kama_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes, self.stoch_k, self.stoch_d,
        )
        self._oi_mom = oi_momentum(self._oi, self.oi_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        k_val = self._kama[bar_index]
        sk = self._stoch_k[bar_index]
        sd = self._stoch_d[bar_index]
        oim = self._oi_mom[bar_index]

        if any(np.isnan(v) for v in (close, k_val, sk, sd)):
            return 0.0

        if close >= k_val:
            return 0.0

        dist = (k_val - close) / k_val
        strength = min(1.0, dist * 35.0)

        signal = -(0.25 + strength * 0.3)

        if sk < sd and sk < 50:
            signal -= 0.2

        if not np.isnan(oim) and oim < 0:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay", "color": "#ffab40"},
            {"name": "%K", "array": self._stoch_k, "type": "subplot", "panel": "Stochastic",
             "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
            {"name": "%D", "array": self._stoch_d, "type": "subplot", "panel": "Stochastic", "style": "dash"},
            {"name": f"OI Mom({self.oi_mom_period})", "array": self._oi_mom, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Stochastic",
                    [
                        self._make_subplot_trace("%K", datetimes, self._stoch_k, color="#bb86fc"),
                        self._make_subplot_trace("%D", datetimes, self._stoch_d, style="dash", color="#78909c"),
                    ],
                    horizontal_lines=[20, 50, 80], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"OI Mom({self.oi_mom_period})",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#ef5350")],
                    zero_line=True,
                ),
            ],
        }
