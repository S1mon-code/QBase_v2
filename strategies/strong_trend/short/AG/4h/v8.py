"""StrongTrendShortAG4hV8 — TEMA Bearish + Stochastic Oversold + OBV Declining.

Economic logic: TEMA reacts fast on 4H — price below TEMA confirms bearish.
Stochastic %K < 20 in trending context means bearish continuation (not reversal).
OBV declining validates cumulative selling volume pressure.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.stochastic import stochastic
from indicators.trend.tema import tema
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV8(TrendingStrategy):
    """TEMA bearish + Stochastic oversold + OBV declining.

    Signal logic:
        close < TEMA AND Stoch_K < 20 AND OBV < OBV_SMA -> -0.85
        close < TEMA AND Stoch_K < 35 -> -0.45
        else -> 0.0
    """

    name = "strong_trend_short_AG_4h_v8"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    tema_period: int = 30
    stoch_k: int = 25
    obv_sma_period: int = 40
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, self.tema_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes,
            k_period=self.stoch_k, d_period=8,
        )
        self._obv = obv(self._closes, self._volumes)
        n = len(self._obv)
        self._obv_sma = np.full(n, np.nan)
        for i in range(self.obv_sma_period - 1, n):
            w = self._obv[i - self.obv_sma_period + 1 : i + 1]
            v = w[~np.isnan(w)]
            if len(v) > 0:
                self._obv_sma[i] = np.mean(v)

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        t = self._tema[bar_index]
        sk = self._stoch_k[bar_index]
        ov = self._obv[bar_index]
        os_ = self._obv_sma[bar_index]

        if np.isnan(c) or np.isnan(t) or np.isnan(sk):
            return 0.0

        if c >= t:
            return 0.0

        obv_declining = not np.isnan(ov) and not np.isnan(os_) and ov < os_

        if sk < 20 and obv_declining:
            return -0.85
        if sk < 35:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay"},
            {"name": "Stoch %K", "array": self._stoch_k, "panel": "Stochastic",
             "y_range": [0, 100], "horizontal_lines": [20, 80]},
            {"name": "Stoch %D", "array": self._stoch_d, "panel": "Stochastic"},
            {"name": "OBV", "array": self._obv, "panel": "OBV"},
            {"name": "OBV SMA", "array": self._obv_sma, "panel": "OBV", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot("Stochastic", [
                self._make_subplot_trace("Stoch %K", datetimes, self._stoch_k, color="#42a5f5"),
                self._make_subplot_trace("Stoch %D", datetimes, self._stoch_d, color="#ff7043"),
            ], horizontal_lines=[20, 80], y_range=[0, 100]),
            self._make_subplot("OBV", [
                self._make_subplot_trace("OBV", datetimes, self._obv, color="#26a69a"),
                self._make_subplot_trace("OBV SMA", datetimes, self._obv_sma, style="dash", color="#ab47bc"),
            ]),
        ]
        return {"overlays": overlays, "subplots": subplots}
