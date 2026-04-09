"""MildTrendLongI1hV4 — ZLEMA(25) + CCI(25) + BSP(20).

Economic logic: ZLEMA reduces lag for responsive 1H iron ore trend detection. CCI
above zero confirms price is above its statistical mean. Buying/selling pressure ratio
validates volume-side conviction with buyers dominating. Signal scales with CCI
magnitude and pressure ratio strength.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.trend.zlema import zlema
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV4(TrendingStrategy):
    name = "long_I_1h_v4"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 35

    zlema_period: int = 25
    cci_period: int = 25
    bsp_period: int = 20
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._zlema = zlema(self._closes, period=self.zlema_period)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._buy_p, self._sell_p, self._pressure_ratio = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes, period=self.bsp_period
        )

    def _generate_signal(self, bar_index: int) -> float:
        z = self._zlema[bar_index]
        z_prev = self._zlema[bar_index - 1] if bar_index > 0 else np.nan
        cci_val = self._cci[bar_index]
        pr = self._pressure_ratio[bar_index]

        if any(np.isnan(v) for v in [z, z_prev, cci_val, pr]):
            return 0.0

        zlema_rising = z > z_prev
        cci_bullish = cci_val > 0.0
        buying_dominant = pr > 1.0

        if not (zlema_rising and cci_bullish and buying_dominant):
            return 0.0

        cci_score = min(1.0, cci_val / 150.0) * 0.4
        pr_score = min(1.0, (pr - 1.0) / 0.5) * 0.3
        return min(1.0, 0.3 + cci_score + pr_score)

    def get_indicator_config(self):
        return [
            {"name": f"ZLEMA({self.zlema_period})", "array": self._zlema, "type": "overlay"},
            {"name": f"CCI({self.cci_period})", "array": self._cci, "type": "subplot", "zero_line": True},
            {"name": "Pressure Ratio", "array": self._pressure_ratio, "type": "subplot", "horizontal_lines": [1.0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"ZLEMA({self.zlema_period})", datetimes, self._zlema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "CCI",
                    [self._make_subplot_trace("CCI", datetimes, self._cci, color="#42a5f5")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Buy/Sell Pressure",
                    [self._make_subplot_trace("Ratio", datetimes, self._pressure_ratio, color="#ab47bc")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
