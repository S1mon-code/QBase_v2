"""StrongTrendShortAGDailyV14 — Bollinger(25,2.0) below middle + CMF(25) < 0.

Economic logic: Price below the Bollinger middle band (SMA 25) signals bearish
bias. CMF(25) negative confirms distribution — selling volume dominates. The
combination filters out false breakdowns where price dips briefly below the
middle band without genuine selling pressure.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV14(TrendingStrategy):
    """Bollinger(25,2.0) price < middle + CMF(25) < 0."""

    name = "strong_trend_short_AG_daily_v14"
    horizon = "slow"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    bb_period: int = 25
    bb_std: float = 2.0
    cmf_period: int = 25
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._bb_upper, self._bb_middle, self._bb_lower = bollinger_bands(
            self._closes, period=self.bb_period, num_std=self.bb_std,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, i: int) -> float:
        c = self._closes[i]
        mid = self._bb_middle[i]
        lower = self._bb_lower[i]
        cf = self._cmf[i]

        if any(np.isnan(v) for v in (c, mid, lower, cf)):
            return 0.0

        if c >= mid or cf >= 0:
            return 0.0

        # How far below middle band
        bb_range = mid - lower
        if bb_range <= 0:
            return 0.0
        depth = min(1.0, (mid - c) / bb_range)

        strength = -(0.35 + depth * 0.35 + min(0.30, abs(cf) * 1.5))
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "BB Upper", "array": self._bb_upper, "type": "overlay"},
            {"name": "BB Middle", "array": self._bb_middle, "type": "overlay"},
            {"name": "BB Lower", "array": self._bb_lower, "type": "overlay"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "panel": "CMF", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("BB Upper", datetimes, self._bb_upper, color="#ef5350"),
                self._make_overlay("BB Middle", datetimes, self._bb_middle, color="#ffab40"),
                self._make_overlay("BB Lower", datetimes, self._bb_lower, color="#66bb6a"),
            ],
            "subplots": [
                self._make_subplot(f"CMF({self.cmf_period})", [
                    self._make_subplot_trace("CMF", datetimes, self._cmf, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
