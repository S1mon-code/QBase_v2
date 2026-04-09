"""StrongTrendLongAG4hV1 — EMA(30,65) + RSI(35) + Twiggs(50).

Economic logic: Dual EMA crossover captures Silver's 4H trend regime. RSI
with wider period filters AG's volatile momentum. Twiggs Money Flow confirms
volume-driven accumulation during uptrends.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.momentum.rsi import rsi
from indicators.volume.twiggs import twiggs_money_flow
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG4hV1(TrendingStrategy):
    name = "strong_trend_long_AG_4h_v1"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 100

    ema_fast: int = 30
    ema_slow: int = 65
    rsi_period: int = 35
    twiggs_period: int = 50
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = ema(self._closes, period=self.ema_fast)
        self._ema_slow = ema(self._closes, period=self.ema_slow)
        self._rsi = rsi(self._closes, period=self.rsi_period)
        self._twiggs = twiggs_money_flow(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.twiggs_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        r = self._rsi[bar_index]
        tw = self._twiggs[bar_index]

        if any(np.isnan(v) for v in (ef, es, r, tw)):
            return 0.0

        if ef > es and r > 50.0 and tw > 0.0:
            strength = min(1.0, (r - 50.0) / 30.0 * 0.5 + tw * 2.0)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast, "type": "overlay"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow, "type": "overlay"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 50, 70]},
            {"name": f"Twiggs({self.twiggs_period})", "array": self._twiggs,
             "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_fast})", datetimes, self._ema_fast, color="#ffab40"),
                self._make_overlay(f"EMA({self.ema_slow})", datetimes, self._ema_slow, color="#ab47bc"),
            ],
            "subplots": [
                self._make_subplot(
                    f"RSI({self.rsi_period})",
                    [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#42a5f5")],
                    y_range=[0, 100], horizontal_lines=[30, 50, 70],
                ),
                self._make_subplot(
                    f"Twiggs({self.twiggs_period})",
                    [self._make_subplot_trace("Twiggs", datetimes, self._twiggs, color="#66bb6a")],
                    zero_line=True,
                ),
            ],
        }
