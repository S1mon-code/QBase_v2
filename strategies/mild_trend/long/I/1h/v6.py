"""MildTrendLongI1hV6 — TEMA(25) + Stochastic(20,5) + ForceIndex(20).

Economic logic: TEMA provides responsive 1H iron ore trend detection with triple
smoothing. Stochastic in the 30-85 zone confirms bullish momentum without overbought
risk. Force Index combines price change with volume for conviction. Signal scales
with stochastic level and force index positivity.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.stochastic import stochastic
from indicators.trend.tema import tema
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV6(TrendingStrategy):
    name = "mild_trend_long_I_1h_v6"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 35

    tema_period: int = 25
    stoch_k: int = 20
    stoch_d: int = 5
    fi_period: int = 20
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, period=self.tema_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes, k_period=self.stoch_k, d_period=self.stoch_d
        )
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        t = self._tema[bar_index]
        t_prev = self._tema[bar_index - 1] if bar_index > 0 else np.nan
        sk = self._stoch_k[bar_index]
        sd = self._stoch_d[bar_index]
        fi_val = self._fi[bar_index]

        if any(np.isnan(v) for v in [t, t_prev, sk, sd, fi_val]):
            return 0.0

        tema_rising = t > t_prev
        stoch_bullish = 30.0 < sk < 85.0 and sk > sd
        fi_positive = fi_val > 0.0

        if not (tema_rising and stoch_bullish and fi_positive):
            return 0.0

        stoch_score = min(1.0, (sk - 30.0) / 50.0) * 0.4
        return min(1.0, 0.3 + stoch_score + 0.2)

    def get_indicator_config(self):
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay"},
            {"name": "Stoch %K", "array": self._stoch_k, "type": "subplot", "panel": "Stochastic",
             "y_range": [0, 100], "horizontal_lines": [20, 80]},
            {"name": "Stoch %D", "array": self._stoch_d, "type": "subplot", "panel": "Stochastic", "style": "dash"},
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Stochastic",
                    [
                        self._make_subplot_trace("%K", datetimes, self._stoch_k, color="#42a5f5"),
                        self._make_subplot_trace("%D", datetimes, self._stoch_d, color="#ff8a80", style="dash"),
                    ],
                    y_range=[0, 100], horizontal_lines=[20, 80],
                ),
                self._make_subplot(
                    "Force Index",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#66bb6a")],
                    zero_line=True,
                ),
            ],
        }
