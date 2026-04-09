"""MildTrendLongIDailyV1 — KAMA(120) + ADX(60) + OBV trend.

Economic logic: KAMA adapts to iron ore's regime-shifting nature — fast in trends,
slow in chop. ADX above 25 confirms a directional market, while rising OBV validates
that volume supports the price move. Signal scales with ADX strength and KAMA slope
magnitude for gradual position sizing.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx
from indicators.trend.kama import kama
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV1(TrendingStrategy):
    name = "long_I_daily_v1"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 150

    # Optimizable parameters (<=5 including chandelier_mult)
    kama_period: int = 120
    adx_period: int = 60
    obv_smooth: int = 120
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._adx = adx(self._highs, self._lows, self._closes, period=self.adx_period)
        self._obv_raw = obv(self._closes, self._volumes)
        # Smooth OBV with simple rolling mean for trend detection
        n = len(self._closes)
        self._obv_ma = np.full(n, np.nan)
        for i in range(self.obv_smooth - 1, n):
            self._obv_ma[i] = np.mean(self._obv_raw[i - self.obv_smooth + 1:i + 1])

    def _generate_signal(self, bar_index: int) -> float:
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1] if bar_index > 0 else np.nan
        adx_val = self._adx[bar_index]
        obv_now = self._obv_raw[bar_index]
        obv_ma = self._obv_ma[bar_index]

        if any(np.isnan(v) for v in [k, k_prev, adx_val, obv_now, obv_ma]):
            return 0.0

        kama_rising = k > k_prev
        adx_strong = adx_val > 25.0
        obv_bullish = obv_now > obv_ma

        if not (kama_rising and adx_strong and obv_bullish):
            return 0.0

        # Scale by ADX strength (25-60 range mapped to 0.3-1.0)
        adx_score = min(1.0, max(0.0, (adx_val - 25.0) / 35.0)) * 0.7 + 0.3
        return min(1.0, adx_score)

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": f"ADX({self.adx_period})", "array": self._adx, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [25]},
            {"name": "OBV", "array": self._obv_raw, "type": "subplot", "panel": "OBV"},
            {"name": "OBV MA", "array": self._obv_ma, "type": "subplot", "panel": "OBV",
             "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"ADX({self.adx_period})",
                    [self._make_subplot_trace("ADX", datetimes, self._adx, color="#42a5f5")],
                    horizontal_lines=[25], y_range=[0, 100],
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv_raw, color="#66bb6a"),
                        self._make_subplot_trace("OBV MA", datetimes, self._obv_ma, color="#ef5350", style="dash"),
                    ],
                ),
            ],
        }
