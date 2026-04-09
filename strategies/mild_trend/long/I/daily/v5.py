"""MildTrendLongIDailyV5 — EMA(100,200) + Donchian(120) + Twiggs(80).

Economic logic: EMA golden cross (100 over 200) captures iron ore's multi-month trend
shifts. Donchian channel breakout confirms price reaching new highs within the trend.
Twiggs Money Flow detects institutional accumulation using true-range-adjusted volume.
Signal strength scales with EMA separation and TMF level.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.trend.ema import ema
from indicators.volume.twiggs import twiggs_money_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV5(TrendingStrategy):
    name = "mild_trend_long_I_daily_v5"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 220

    ema_fast: int = 100
    ema_slow: int = 200
    dc_period: int = 120
    tmf_period: int = 80
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = ema(self._closes, period=self.ema_fast)
        self._ema_slow = ema(self._closes, period=self.ema_slow)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.dc_period
        )
        self._tmf = twiggs_money_flow(
            self._highs, self._lows, self._closes, self._volumes, period=self.tmf_period
        )

    def _generate_signal(self, bar_index: int) -> float:
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        dc_up = self._dc_upper[bar_index]
        tmf = self._tmf[bar_index]
        close = self._closes[bar_index]

        if any(np.isnan(v) for v in [ef, es, dc_up, tmf, close]):
            return 0.0

        golden_cross = ef > es
        near_dc_high = close > dc_up * 0.98 if dc_up > 0 else False
        tmf_positive = tmf > 0.0

        if not (golden_cross and near_dc_high and tmf_positive):
            return 0.0

        # EMA separation as signal strength
        sep = (ef - es) / es if es > 0 else 0.0
        ema_score = min(1.0, sep / 0.05) * 0.5
        tmf_score = min(1.0, tmf / 0.2) * 0.3
        return min(1.0, 0.2 + ema_score + tmf_score)

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast, "type": "overlay"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow, "type": "overlay"},
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "style": "dash"},
            {"name": f"TMF({self.tmf_period})", "array": self._tmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_fast})", datetimes, self._ema_fast, color="#ffab40"),
                self._make_overlay(f"EMA({self.ema_slow})", datetimes, self._ema_slow, color="#ab47bc"),
                self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, style="dash", color="#78909c"),
            ],
            "subplots": [
                self._make_subplot(
                    "Twiggs MF",
                    [self._make_subplot_trace("TMF", datetimes, self._tmf, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
