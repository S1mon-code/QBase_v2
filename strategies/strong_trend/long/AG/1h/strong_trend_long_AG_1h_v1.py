"""StrongTrendLongAG1hV1 — EMA(20,50) + RSI(20) + VolumeSpike(15).

Economic logic: Fast EMA crossover captures Silver's explosive 1H breakouts.
RSI with short period detects momentum surges. Volume spike detection confirms
institutional participation in AG's fast moves.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.momentum.rsi import rsi
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG1hV1(TrendingStrategy):
    name = "strong_trend_long_AG_1h_v1"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 20
    vol_spike_period: int = 15
    chandelier_mult: float = 2.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = ema(self._closes, period=self.ema_fast)
        self._ema_slow = ema(self._closes, period=self.ema_slow)
        self._rsi = rsi(self._closes, period=self.rsi_period)
        self._vol_spike = volume_spike(
            self._volumes, period=self.vol_spike_period, threshold=2.0,
        )

    def _generate_signal(self, bar_index: int) -> float:
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        r = self._rsi[bar_index]

        if any(np.isnan(v) for v in (ef, es, r)):
            return 0.0

        spike = bool(self._vol_spike[bar_index]) if bar_index < len(self._vol_spike) else False

        if ef > es and r > 50.0:
            base = min(1.0, (r - 50.0) / 25.0 * 0.5 + 0.3)
            if spike:
                base = min(1.0, base + 0.25)
            return max(0.0, base)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast, "type": "overlay"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow, "type": "overlay"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi,
             "type": "subplot", "y_range": [0, 100], "horizontal_lines": [30, 50, 70]},
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
            ],
        }
