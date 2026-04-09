"""MildTrendLongI4hV16 — Aroon + CMF + ADX (3-way confirmation).

Economic logic: Three independent dimensions — Aroon measures recency of highs/lows,
CMF measures buying/selling pressure via volume-weighted close location, and ADX
filters out ranging markets.  Iron ore trends are strongest when all three dimensions
align.  Two-of-three agreement provides a moderate signal.  Low ADX kills the signal
entirely to avoid whipsaw losses.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.trend.aroon import aroon
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV16(TrendingStrategy):
    """Three-way trend confirmation: Aroon oscillator + CMF + ADX filter.

    Signal logic:
        Aroon osc > 50 AND CMF > 0 AND ADX > 20 → +min(1.0, aroon_osc/100)
        All 3 bearish                             → -min(1.0, abs(aroon_osc)/100)
        2-of-3 agree                              → 0.4 * direction
        ADX < 20                                  → 0.0
    """

    name = "mild_trend_long_I_4h_v16"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 45  # max(aroon, cmf, adx) + 20

    # Optimizable parameters
    aroon_period: int = 25
    cmf_period: int = 20
    adx_period: int = 14
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
        """Precompute Aroon, CMF, and ADX arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        _, _, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period,
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )
        self._adx, _, _ = adx_with_di(
            self._highs, self._lows, self._closes, period=self.adx_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from 3-way Aroon + CMF + ADX confirmation."""
        aroon_osc = self._aroon_osc[bar_index]
        cmf_val = self._cmf[bar_index]
        adx_val = self._adx[bar_index]

        if np.isnan(aroon_osc) or np.isnan(cmf_val) or np.isnan(adx_val):
            return 0.0

        if adx_val < 20.0:
            return 0.0

        aroon_bull = aroon_osc > 50.0
        aroon_bear = aroon_osc < -50.0
        cmf_bull = cmf_val > 0.0
        cmf_bear = cmf_val < 0.0

        # All three bullish
        if aroon_bull and cmf_bull:
            return min(1.0, aroon_osc / 100.0)
        # All three bearish
        if aroon_bear and cmf_bear:
            return -min(1.0, abs(aroon_osc) / 100.0)

        # 2-of-3 agreement
        bull_count = int(aroon_osc > 0.0) + int(cmf_bull)
        bear_count = int(aroon_osc < 0.0) + int(cmf_bear)

        if bull_count >= 2:
            return 0.4
        if bear_count >= 2:
            return -0.4

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "aroon", "params": {"period": self.aroon_period}},
            {"name": "cmf", "params": {"period": self.cmf_period}},
            {"name": "adx", "params": {"period": self.adx_period}},
        ]
