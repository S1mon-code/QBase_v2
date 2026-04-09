"""MildTrendLongI4hV8 — TEMA(35) + Aroon(40) + VolumeSpike(30).

Economic logic: TEMA's triple exponential smoothing provides responsive 4H iron ore
trend tracking. Aroon Up above Down with Up > 50 confirms bullish trend maturity.
Volume spikes validate conviction behind breakout moves. Signal scales with Aroon
up-down differential and is boosted during volume spikes.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.trend.tema import tema
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV8(TrendingStrategy):
    name = "mild_trend_long_I_4h_v8"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 50

    tema_period: int = 35
    aroon_period: int = 40
    vs_period: int = 30
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, period=self.tema_period)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period
        )
        self._vol_spike = volume_spike(self._volumes, period=self.vs_period)

    def _generate_signal(self, bar_index: int) -> float:
        t = self._tema[bar_index]
        t_prev = self._tema[bar_index - 1] if bar_index > 0 else np.nan
        ar_up = self._aroon_up[bar_index]
        ar_down = self._aroon_down[bar_index]

        if any(np.isnan(v) for v in [t, t_prev, ar_up, ar_down]):
            return 0.0

        tema_rising = t > t_prev
        aroon_bullish = ar_up > ar_down and ar_up > 50.0

        if not (tema_rising and aroon_bullish):
            return 0.0

        aroon_score = min(1.0, (ar_up - 50.0) / 50.0) * 0.5
        spike_bonus = 0.15 if self._vol_spike[bar_index] else 0.0
        return min(1.0, 0.3 + aroon_score + spike_bonus)

    def get_indicator_config(self):
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay"},
            {"name": "Aroon Up", "array": self._aroon_up, "type": "subplot", "panel": "Aroon"},
            {"name": "Aroon Down", "array": self._aroon_down, "type": "subplot", "panel": "Aroon"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Aroon",
                    [
                        self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#66bb6a"),
                        self._make_subplot_trace("Down", datetimes, self._aroon_down, color="#ef5350"),
                    ],
                    y_range=[0, 100], horizontal_lines=[50],
                ),
            ],
        }
