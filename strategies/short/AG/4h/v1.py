"""StrongTrendShortAG4hV1 — ADX Trend + DI- Dominance + OBV Distribution.

Economic logic: ADX(25) > 22 confirms a strong directional trend on 4H silver.
DI-(25) > DI+(25) validates that sellers dominate. OBV falling below its SMA(40)
confirms sustained distribution — money is leaving the market. The slope-based
OBV filter and high ADX threshold prevent whipsaw entries in choppy conditions.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.trend.ema import ema
from indicators.trend.sma import sma
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV1(TrendingStrategy):
    """ADX trend strength + DI- dominance + OBV distribution.

    Signal logic (EMA-smoothed, period=4):
        ADX > 22 AND DI- > DI+ AND OBV < OBV_SMA(40) -> -0.90
        ADX > 22 AND DI- > DI+ -> -0.55
        else -> 0.0
    """

    name = "short_AG_4h_v1"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    adx_period: int = 25
    adx_threshold: float = 22.0
    obv_sma_period: int = 40
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._adx, self._di_plus, self._di_minus = adx_with_di(
            self._highs, self._lows, self._closes, period=self.adx_period,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, period=self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        ax = self._adx[i]
        dp = self._di_plus[i]
        dm = self._di_minus[i]
        ov = self._obv[i]
        os_ = self._obv_sma[i]

        if np.isnan(ax) or np.isnan(dp) or np.isnan(dm):
            return 0.0
        if ax <= self.adx_threshold or dm <= dp:
            return 0.0
        if not np.isnan(os_) and ov < os_:
            return -0.90
        return -0.55

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "ADX", "array": self._adx, "panel": "ADX/DI",
             "y_range": [0, 100], "horizontal_lines": [self.adx_threshold]},
            {"name": "DI+", "array": self._di_plus, "panel": "ADX/DI", "color": "#26a69a"},
            {"name": "DI-", "array": self._di_minus, "panel": "ADX/DI", "color": "#ef5350"},
            {"name": "OBV", "array": self._obv, "panel": "OBV", "zero_line": True},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma,
             "panel": "OBV", "color": "#ff7043"},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = []
        subplots = [
            self._make_subplot("ADX / DI", [
                self._make_subplot_trace("ADX", datetimes, self._adx, color="#42a5f5"),
                self._make_subplot_trace("DI+", datetimes, self._di_plus, color="#26a69a"),
                self._make_subplot_trace("DI-", datetimes, self._di_minus, color="#ef5350"),
            ], y_range=[0, 100], horizontal_lines=[self.adx_threshold]),
            self._make_subplot("OBV", [
                self._make_subplot_trace("OBV", datetimes, self._obv, color="#42a5f5"),
                self._make_subplot_trace(f"OBV SMA({self.obv_sma_period})", datetimes,
                                         self._obv_sma, color="#ff7043"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
