"""StrongTrendShortAG2hV5 — KAMA Slope + Aroon Down Dominant + Force Index Negative.

Economic logic: KAMA(25) slope negative adapts to volatility and confirms
persistent downtrend. Aroon Down(25) > 65 means recent lows dominate the
lookback window. Force Index(15) < 0 confirms bearish volume-price force.
EMA(4) smoothing prevents overtrading.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.kama import kama
from indicators.trend.aroon import aroon
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV5(TrendingStrategy):
    """KAMA(25) slope negative + Aroon Down(25) > 65 + Force Index(15) < 0.

    Signal logic (raw, pre-smoothing):
        All 3 conditions met -> -0.85
        KAMA slope neg AND Aroon Down > 65 -> -0.55
        KAMA slope neg AND Force Index < 0 -> -0.35
        else -> 0.0
    """

    name = "short_AG_2h_v5"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    kama_period: int = 25
    aroon_period: int = 25
    fi_period: int = 15
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period,
        )
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        k = self._kama[i]
        k_prev = self._kama[i - 1] if i > 0 else np.nan
        ad = self._aroon_down[i]
        fi = self._fi[i]

        if any(np.isnan(v) for v in (k, k_prev, ad, fi)):
            return 0.0

        kama_slope_neg = k < k_prev
        aroon_down_dom = ad > 65.0
        fi_neg = fi < 0.0

        if kama_slope_neg and aroon_down_dom and fi_neg:
            return -0.85
        if kama_slope_neg and aroon_down_dom:
            return -0.55
        if kama_slope_neg and fi_neg:
            return -0.35
        return 0.0

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama,
             "type": "overlay"},
            {"name": f"Aroon Down({self.aroon_period})", "array": self._aroon_down,
             "panel": "Aroon", "horizontal_lines": [65]},
            {"name": f"Aroon Up({self.aroon_period})", "array": self._aroon_up,
             "panel": "Aroon"},
            {"name": f"Force Index({self.fi_period})", "array": self._fi,
             "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"KAMA({self.kama_period})", datetimes,
                               self._kama, color="#ffab40"),
        ]
        subplots = [
            self._make_subplot("Aroon", [
                self._make_subplot_trace("Aroon Up", datetimes, self._aroon_up,
                                         color="#66bb6a"),
                self._make_subplot_trace("Aroon Down", datetimes, self._aroon_down,
                                         color="#ef5350"),
            ], horizontal_lines=[65], y_range=[0, 100]),
            self._make_subplot(f"Force Index({self.fi_period})", [
                self._make_subplot_trace("FI", datetimes, self._fi, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
