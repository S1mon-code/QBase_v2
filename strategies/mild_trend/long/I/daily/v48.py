"""MildTrendLongIDailyV48 — CCI(25) > 0 + OI momentum(20) > 0 + EMA(30) slope up.

Economic logic: CCI measures deviation from statistical mean — positive CCI shows
price above its average. Open interest momentum confirms new money entering longs.
Rising EMA validates the underlying trend direction. Triple confirmation from price,
positioning, and trend.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.volume.oi_momentum import oi_momentum
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV48(TrendingStrategy):
    name = "mild_trend_long_I_daily_v48"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 60

    cci_period: int = 25
    oi_period: int = 20
    ema_period: int = 30
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._oi_mom = oi_momentum(self._oi, period=self.oi_period)
        self._ema = ema(self._closes, period=self.ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        cci_val = self._cci[bar_index]
        oi_val = self._oi_mom[bar_index]
        e = self._ema[bar_index]
        e_prev = self._ema[bar_index - 1] if bar_index > 0 else np.nan

        if any(np.isnan(v) for v in [cci_val, oi_val, e, e_prev]):
            return 0.0

        cci_positive = cci_val > 0.0
        oi_bullish = oi_val > 0.0
        ema_rising = e > e_prev

        if not (cci_positive and oi_bullish and ema_rising):
            return 0.0

        # Scale by CCI magnitude (0-200 mapped to 0.3-0.8)
        cci_score = min(0.5, max(0.0, cci_val / 400.0)) + 0.3
        # Boost with OI momentum
        oi_boost = min(0.2, max(0.0, oi_val * 0.5))
        return min(1.0, cci_score + oi_boost)

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_period})", "array": self._ema, "type": "overlay"},
            {"name": f"CCI({self.cci_period})", "array": self._cci, "type": "subplot",
             "horizontal_lines": [0, 100, -100]},
            {"name": f"OI Mom({self.oi_period})", "array": self._oi_mom, "type": "subplot",
             "horizontal_lines": [0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_period})", datetimes, self._ema,
                                   color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"CCI({self.cci_period})",
                    [self._make_subplot_trace("CCI", datetimes, self._cci, color="#42a5f5")],
                    horizontal_lines=[0, 100, -100],
                ),
                self._make_subplot(
                    f"OI Momentum({self.oi_period})",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom,
                                             color="#66bb6a")],
                    horizontal_lines=[0],
                ),
            ],
        }
