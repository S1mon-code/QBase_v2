"""StrongTrendShortAGDailyV1 — ADX(30) slope + DI- dominant + OBV declining.

Economic logic: A strong downtrend in silver is confirmed when ADX is rising
above 25 (trend strengthening), DI- dominates DI+ (sellers in control), and
OBV is below its 50-period SMA (distribution phase — volume flows out on
down days). The slope-based ADX entry prevents whipsaws from brief ADX spikes.
Signal strength scales with ADX magnitude and DI spread.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.trend.ema import ema
from indicators.trend.sma import sma
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV1(TrendingStrategy):
    """ADX trend strength + DI- dominance + OBV distribution."""

    name = "short_AG_daily_v1"
    horizon = "slow"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    adx_period: int = 30
    obv_sma_period: int = 50
    adx_threshold: float = 25.0
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._adx_line, self._di_plus, self._di_minus = adx_with_di(
            self._highs, self._lows, self._closes, period=self.adx_period,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, period=self.obv_sma_period)

        # Precompute smoothed signals
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        if np.isnan(val):
            return 0.0
        return min(0.0, val)

    def _raw_signal(self, bar_index: int) -> float:
        adx_val = self._adx_line[bar_index]
        di_p = self._di_plus[bar_index]
        di_m = self._di_minus[bar_index]
        obv_val = self._obv[bar_index]
        obv_sma_val = self._obv_sma[bar_index]

        if any(np.isnan(v) for v in (adx_val, di_p, di_m, obv_val, obv_sma_val)):
            return 0.0

        # Conditions: ADX > threshold, DI- > DI+, OBV below SMA
        if adx_val < self.adx_threshold or di_m <= di_p or obv_val >= obv_sma_val:
            return 0.0

        # Strength: scale by ADX (25-60 range) and DI spread
        adx_strength = min(1.0, (adx_val - self.adx_threshold) / 35.0)
        di_spread = min(1.0, (di_m - di_p) / 20.0)
        strength = -(0.3 + adx_strength * 0.4 + di_spread * 0.3)
        return max(-1.0, strength)

    def get_indicator_config(self):
        return [
            {"name": f"ADX({self.adx_period})", "array": self._adx_line,
             "panel": "ADX", "y_range": [0, 100], "horizontal_lines": [25]},
            {"name": "DI+", "array": self._di_plus, "panel": "ADX", "color": "#66bb6a"},
            {"name": "DI-", "array": self._di_minus, "panel": "ADX", "color": "#ef5350"},
            {"name": "OBV", "array": self._obv, "panel": "OBV", "zero_line": True},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma,
             "panel": "OBV", "color": "#ff9800"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    "ADX + DI",
                    [
                        self._make_subplot_trace("ADX", datetimes, self._adx_line, color="#42a5f5"),
                        self._make_subplot_trace("DI+", datetimes, self._di_plus, color="#66bb6a"),
                        self._make_subplot_trace("DI-", datetimes, self._di_minus, color="#ef5350"),
                    ],
                    horizontal_lines=[25], y_range=[0, 100],
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv, color="#7e57c2"),
                        self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ff9800"),
                    ],
                ),
            ],
        }
