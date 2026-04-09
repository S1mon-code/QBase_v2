"""StrongTrendShortAGDailyV3 — KAMA Bearish + Coppock Negative + CMF Outflow.

Economic logic: KAMA adapts to volatility — when price falls below KAMA the trend
is confirmed bearish. Coppock curve turning negative signals long-term momentum
breakdown. CMF < 0 confirms money is leaving silver.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.coppock import coppock
from indicators.trend.kama import kama
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV3(TrendingStrategy):
    """KAMA bearish trend + Coppock negative + CMF outflow.

    Signal logic:
        close < KAMA AND Coppock < 0 AND CMF < 0 -> -0.85
        close < KAMA AND Coppock < 0 -> -0.45
        else -> 0.0
    """

    name = "short_AG_daily_v3"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 250

    kama_period: int = 130
    coppock_wma: int = 120
    cmf_period: int = 100
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._coppock = coppock(self._closes, wma_period=self.coppock_wma, roc_long=180, roc_short=60)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        k = self._kama[bar_index]
        cop = self._coppock[bar_index]
        cf = self._cmf[bar_index]

        if np.isnan(c) or np.isnan(k) or np.isnan(cop):
            return 0.0

        if c >= k:
            return 0.0

        if cop < 0 and (not np.isnan(cf) and cf < 0):
            return -0.85
        if cop < 0:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": f"Coppock({self.coppock_wma})", "array": self._coppock, "zero_line": True},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot(f"Coppock({self.coppock_wma})", [
                self._make_subplot_trace("Coppock", datetimes, self._coppock, color="#42a5f5"),
            ], zero_line=True),
            self._make_subplot(f"CMF({self.cmf_period})", [
                self._make_subplot_trace("CMF", datetimes, self._cmf, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
