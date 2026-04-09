"""Triple EMA Alignment + Force Index — Trend stacking with volume-weighted momentum.

Three EMAs of increasing length form a "bullish stack" when fast > mid > slow,
confirming a multi-timeframe trend alignment on the 1H iron ore chart.
The Force Index adds a volume-weighted momentum dimension, ensuring the trend
is backed by real participation.  Signal strength scales with force index
magnitude relative to its recent peak.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV20(TrendingStrategy):
    """Long signal on triple-EMA bullish stack with positive Force Index.

    Signal logic
    ------------
    * ema_fast > ema_mid > ema_slow AND force_index > 0
        ->  min(1.0, 0.6 + abs(fi) / rolling_max_abs_fi * 0.4)
    * ema_fast > ema_mid (partial stack)
        ->  0.2
    * else  ->  0.0
    """

    name = "long_I_1h_v20"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    fast_period: int = 15
    mid_period: int = 35
    slow_period: int = 70
    fi_period: int = 15
    chandelier_mult: float = 2.5

    def on_init_arrays(
        self, closes, highs, lows, opens, volumes, oi, datetimes
    ):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = ema(closes, self.fast_period)
        self._ema_mid = ema(closes, self.mid_period)
        self._ema_slow = ema(closes, self.slow_period)
        self._fi = force_index(closes, volumes, self.fi_period)

        # Precompute rolling max of abs(force_index) over 50 bars
        abs_fi = np.abs(self._fi)
        n = len(abs_fi)
        self._rolling_max_fi = np.full(n, np.nan)
        window = 50
        for i in range(n):
            start = max(0, i - window + 1)
            self._rolling_max_fi[i] = np.nanmax(abs_fi[start : i + 1])

    def _generate_signal(self, bar_index: int) -> float:
        ema_f = self._ema_fast[bar_index]
        ema_m = self._ema_mid[bar_index]
        ema_s = self._ema_slow[bar_index]
        fi_val = self._fi[bar_index]
        rolling_max = self._rolling_max_fi[bar_index]

        if ema_f > ema_m > ema_s and fi_val > 0:
            ratio = abs(fi_val) / rolling_max if rolling_max > 0 else 0.0
            return min(1.0, 0.6 + ratio * 0.4)
        if ema_f > ema_m:
            return 0.2
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "EMA Fast", "array": self._ema_fast},
            {"name": "EMA Mid", "array": self._ema_mid},
            {"name": "EMA Slow", "array": self._ema_slow},
            {"name": "Force Index", "array": self._fi},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"EMA({self.fast_period})", datetimes, self._ema_fast, color="#ffab40"),
            self._make_overlay(f"EMA({self.mid_period})", datetimes, self._ema_mid, color="#ab47bc"),
            self._make_overlay(f"EMA({self.slow_period})", datetimes, self._ema_slow, color="#4fc3f7")
        ]
        subplots = [
            self._make_subplot(
                f"Force Index({self.fi_period})",
                [self._make_subplot_trace("Force Index", datetimes, self._fi, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}

