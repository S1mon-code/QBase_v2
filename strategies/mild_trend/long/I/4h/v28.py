"""MildTrendLongI4hV28 — Decycler Trend + CMF Confirmation.

Economic logic: Ehlers Decycler removes cyclical components from price, leaving
the underlying trend. In iron ore, which exhibits strong supply-driven cycles,
stripping out the cycle reveals the fundamental trend. CMF confirms volume is
flowing with that trend.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.decycler import decycler
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV28(TrendingStrategy):
    """Price vs Decycler trend confirmed by Chaikin Money Flow.

    Signal logic:
        - Close > decycler AND CMF > 0 -> +1.0
        - Close < decycler AND CMF < 0 -> -1.0
        - Price agrees but no CMF confirm -> +/-0.4
        - Disagree -> 0.0
    """

    name = "mild_trend_long_I_4h_v28"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 100  # decycler_period(60) + cmf_period(20) + 20

    decycler_period: int = 60
    cmf_period: int = 20
    chandelier_mult: float = 2.5

    def on_init_arrays(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        volumes: np.ndarray,
        oi: np.ndarray,
        datetimes: np.ndarray,
    ) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._decycler = decycler(self._closes, period=self.decycler_period)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc = self._decycler[bar_index]
        cv = self._cmf[bar_index]

        if np.isnan(close) or np.isnan(dc) or np.isnan(cv):
            return 0.0

        price_above = close > dc
        price_below = close < dc

        if price_above and cv > 0:
            return 1.0
        if price_below and cv < 0:
            return -1.0
        if price_above:
            return 0.4
        if price_below:
            return -0.4
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "decycler", "params": {"period": self.decycler_period}},
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]
