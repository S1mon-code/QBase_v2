"""StrongTrendShortAGDailyV11 — EMA(40,100) bearish + Williams %R(20) + OBV declining.

Economic logic: EMA(40) < EMA(100) confirms a sustained daily downtrend in silver.
Williams %R(20) < -50 indicates price is trading in the lower half of the recent
range, confirming bearish momentum. OBV below its SMA validates distribution —
institutional selling pressure drives volume-weighted outflows.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.williams_r import williams_r
from indicators.trend.ema import ema
from indicators.trend.sma import sma
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV11(TrendingStrategy):
    """EMA(40) < EMA(100) + Williams %R(20) < -50 + OBV < SMA(50)."""

    name = "strong_trend_short_AG_daily_v11"
    horizon = "slow"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 110

    ema_fast: int = 40
    ema_slow: int = 100
    wr_period: int = 20
    obv_sma_period: int = 50
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = ema(self._closes, self.ema_fast)
        self._ema_slow = ema(self._closes, self.ema_slow)
        self._wr = williams_r(self._highs, self._lows, self._closes, self.wr_period)
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, period=self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, i: int) -> float:
        ef = self._ema_fast[i]
        es = self._ema_slow[i]
        wr = self._wr[i]
        ov = self._obv[i]
        os_ = self._obv_sma[i]

        if any(np.isnan(v) for v in (ef, es, wr, ov, os_)):
            return 0.0

        if ef >= es or wr >= -50.0 or ov >= os_:
            return 0.0

        strength = -0.40
        if wr < -70.0:
            strength -= 0.25
        if ef < es * 0.98:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast, "type": "overlay"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow, "type": "overlay"},
            {"name": f"Williams %R({self.wr_period})", "array": self._wr,
             "panel": "WR", "y_range": [-100, 0], "horizontal_lines": [-50, -80]},
            {"name": "OBV", "array": self._obv, "panel": "OBV"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma, "panel": "OBV"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_fast})", datetimes, self._ema_fast, color="#ffab40"),
                self._make_overlay(f"EMA({self.ema_slow})", datetimes, self._ema_slow, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(f"Williams %R({self.wr_period})", [
                    self._make_subplot_trace("WR", datetimes, self._wr, color="#bb86fc"),
                ], horizontal_lines=[-50, -80], y_range=[-100, 0]),
                self._make_subplot("OBV", [
                    self._make_subplot_trace("OBV", datetimes, self._obv, color="#7e57c2"),
                    self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ff9800"),
                ]),
            ],
        }
