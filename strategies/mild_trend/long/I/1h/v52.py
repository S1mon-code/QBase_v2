"""MildTrendLongI1hV52 — Fisher Transform(10) bullish + OBV above SMA(20).

Economic logic: Fisher Transform on 1h bars converts price to Gaussian distribution
for sharper turning point detection. OBV above its SMA confirms accumulation phase.
Signal scales with Fisher spread and is boosted on fresh bullish crosses.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.obv import obv
from indicators.trend.sma import sma
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV52(TrendingStrategy):
    name = "mild_trend_long_I_1h_v52"
    horizon = "fast"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 40

    fisher_period: int = 10
    obv_sma_period: int = 20
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._fisher, self._fisher_trigger = fisher_transform(
            self._highs, self._lows, period=self.fisher_period)
        self._obv_raw = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv_raw, period=self.obv_sma_period)

    def _generate_signal(self, bar_index: int) -> float:
        f_val = self._fisher[bar_index]
        f_trig = self._fisher_trigger[bar_index]
        f_prev = self._fisher[bar_index - 1] if bar_index > 0 else np.nan
        ft_prev = self._fisher_trigger[bar_index - 1] if bar_index > 0 else np.nan
        obv_now = self._obv_raw[bar_index]
        obv_ma = self._obv_sma[bar_index]

        if any(np.isnan(v) for v in [f_val, f_trig, f_prev, ft_prev, obv_now, obv_ma]):
            return 0.0

        bullish_cross = (f_prev <= ft_prev) and (f_val > f_trig)
        fisher_above = f_val > f_trig
        obv_bullish = obv_now > obv_ma

        if not (fisher_above and obv_bullish):
            return 0.0

        base = 0.5 if bullish_cross else 0.3
        fisher_spread = min(0.5, max(0.0, (f_val - f_trig) * 0.25))
        return min(1.0, base + fisher_spread)

    def get_indicator_config(self):
        return [
            {"name": "Fisher", "array": self._fisher, "type": "subplot"},
            {"name": "Trigger", "array": self._fisher_trigger, "type": "subplot", "style": "dash"},
            {"name": "OBV", "array": self._obv_raw, "type": "subplot", "panel": "OBV"},
            {"name": "OBV SMA", "array": self._obv_sma, "type": "subplot", "panel": "OBV",
             "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"Fisher({self.fisher_period})",
                    [
                        self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#42a5f5"),
                        self._make_subplot_trace("Trigger", datetimes, self._fisher_trigger,
                                                 color="#ef5350", style="dash"),
                    ],
                    horizontal_lines=[0],
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv_raw, color="#66bb6a"),
                        self._make_subplot_trace("OBV SMA", datetimes, self._obv_sma,
                                                 color="#ff7043", style="dash"),
                    ],
                ),
            ],
        }
