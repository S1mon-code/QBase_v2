"""StrongTrendShortAG2hV1 — EMA Slope + ADX/DI Bearish + CMF Negative.

Economic logic: EMA(30) slope confirms sustained downtrend direction.
ADX(20) > 22 with DI- > DI+ confirms strong bearish momentum.
CMF(20) < 0 validates distribution (selling pressure exceeds buying).
Signal EMA(4) smoothing prevents whipsaw entries in volatile downtrends.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.adx import adx_with_di
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV1(TrendingStrategy):
    """EMA(30) slope < 0 + ADX(20) > 22 + DI- > DI+ + CMF(20) < 0.

    Signal logic (raw, pre-smoothing):
        All 4 conditions met -> -0.90
        EMA slope < 0 AND ADX > 22 AND DI- > DI+ -> -0.55
        EMA slope < 0 AND CMF < 0 -> -0.30
        else -> 0.0
    """

    name = "short_AG_2h_v1"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 65

    ema_period: int = 30
    adx_period: int = 20
    cmf_period: int = 20
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_line = ema(self._closes, period=self.ema_period)
        self._adx, self._plus_di, self._minus_di = adx_with_di(
            self._highs, self._lows, self._closes, period=self.adx_period,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes,
                        period=self.cmf_period)

        # Precompute raw signals then smooth
        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        e = self._ema_line[i]
        e_prev = self._ema_line[i - 1]
        a = self._adx[i]
        pdi = self._plus_di[i]
        mdi = self._minus_di[i]
        c = self._cmf[i]

        if any(np.isnan(v) for v in (e, e_prev, a, pdi, mdi, c)):
            return 0.0

        slope_neg = e < e_prev
        adx_strong = a > 22.0
        di_bearish = mdi > pdi
        cmf_neg = c < 0.0

        if slope_neg and adx_strong and di_bearish and cmf_neg:
            return -0.90
        if slope_neg and adx_strong and di_bearish:
            return -0.55
        if slope_neg and cmf_neg:
            return -0.30
        return 0.0

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"EMA({self.ema_period})", "array": self._ema_line,
             "type": "overlay"},
            {"name": f"ADX({self.adx_period})", "array": self._adx,
             "y_range": [0, 100], "horizontal_lines": [22]},
            {"name": "+DI", "array": self._plus_di, "panel": f"ADX({self.adx_period})"},
            {"name": "-DI", "array": self._minus_di, "panel": f"ADX({self.adx_period})"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"EMA({self.ema_period})", datetimes,
                               self._ema_line, color="#ffab40"),
        ]
        subplots = [
            self._make_subplot(f"ADX({self.adx_period})", [
                self._make_subplot_trace("ADX", datetimes, self._adx, color="#42a5f5"),
                self._make_subplot_trace("+DI", datetimes, self._plus_di, color="#66bb6a"),
                self._make_subplot_trace("-DI", datetimes, self._minus_di, color="#ef5350"),
            ], horizontal_lines=[22], y_range=[0, 100]),
            self._make_subplot(f"CMF({self.cmf_period})", [
                self._make_subplot_trace("CMF", datetimes, self._cmf, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
