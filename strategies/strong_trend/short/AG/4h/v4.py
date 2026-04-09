"""StrongTrendShortAG4hV4 — KAMA Bearish + CCI Oversold + CMF Outflow.

Economic logic: KAMA adapts to noise — price below KAMA45 on 4H shows the adaptive
trend is bearish. CCI < -100 signals strong bearish momentum for AG.
CMF < 0 confirms money is leaving silver.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.trend.kama import kama
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV4(TrendingStrategy):
    """KAMA bearish + CCI oversold + CMF outflow.

    Signal logic:
        close < KAMA AND CCI < -100 AND CMF < 0 -> -0.85
        close < KAMA AND CCI < -75 -> -0.45
        else -> 0.0
    """

    name = "strong_trend_short_AG_4h_v4"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    kama_period: int = 45
    cci_period: int = 30
    cmf_period: int = 35
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        k = self._kama[bar_index]
        cc = self._cci[bar_index]
        cf = self._cmf[bar_index]

        if np.isnan(c) or np.isnan(k) or np.isnan(cc):
            return 0.0

        if c >= k:
            return 0.0

        if cc < -100 and (not np.isnan(cf) and cf < 0):
            return -0.85
        if cc < -75:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": f"CCI({self.cci_period})", "array": self._cci,
             "zero_line": True, "horizontal_lines": [-100, 100]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot(f"CCI({self.cci_period})", [
                self._make_subplot_trace("CCI", datetimes, self._cci, color="#42a5f5"),
            ], zero_line=True, horizontal_lines=[-100, 100]),
            self._make_subplot(f"CMF({self.cmf_period})", [
                self._make_subplot_trace("CMF", datetimes, self._cmf, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
