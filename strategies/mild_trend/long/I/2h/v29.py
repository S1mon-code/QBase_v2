"""MildTrendLongI2hV29 — ADX + ROC + CMF (3-way confirmation).

Economic logic: ADX measures trend strength (>25 = trending), ROC captures
directional momentum, and CMF confirms volume-weighted money flow. A 3-way
agreement produces high-conviction signals; 2-of-3 agreement yields a
reduced-strength signal. ADX < 25 filters out ranging markets.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.momentum.roc import rate_of_change
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV29(TrendingStrategy):
    """ADX trend filter with ROC and CMF 3-way confirmation.

    Signal logic:
        - ADX > 25 AND ROC > 0 AND CMF > 0: +min(1.0, ADX/50)
        - ADX > 25 AND ROC < 0 AND CMF < 0: -min(1.0, ADX/50)
        - ADX > 25, 2-of-3 agree: 0.4 * direction
        - ADX < 25: 0.0
    """

    name = "mild_trend_long_I_2h_v29"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 40  # max(adx_period(14), roc_period(14), cmf_period(20)) + 20

    adx_period: int = 14
    roc_period: int = 14
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
        """Precompute ADX, ROC, and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        adx_arr, _, _ = adx_with_di(
            self._highs, self._lows, self._closes, period=self.adx_period,
        )
        self._adx = adx_arr
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on ADX, ROC, and CMF."""
        a = self._adx[bar_index]
        r = self._roc[bar_index]
        c = self._cmf[bar_index]

        if np.isnan(a) or np.isnan(r) or np.isnan(c):
            return 0.0

        if a < 25:
            return 0.0

        strength = float(np.clip(a / 50.0, 0.0, 1.0))
        roc_bull = r > 0
        cmf_bull = c > 0

        # 3-way agreement
        if roc_bull and cmf_bull:
            return strength
        if not roc_bull and not cmf_bull:
            return -strength

        # 2-of-3 agreement (ADX confirms trend exists, one indicator gives direction)
        if roc_bull:
            return 0.4
        return -0.4

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "adx", "params": {"period": self.adx_period}},
            {"name": "roc", "params": {"period": self.roc_period}},
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]
