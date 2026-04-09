"""MildTrendLongI4hV12 — ADX + MACD Line Direction.

Economic logic: ADX measures trend strength without directional bias while DI lines
reveal who controls the market.  MACD line provides momentum confirmation.  When ADX
is high (strong trend) and DI + MACD agree on direction, the trend is robust.  Low
ADX signals a ranging market where trend-following loses edge.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.adx import adx_with_di
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV12(TrendingStrategy):
    """ADX trend strength with DI direction confirmed by MACD line.

    Signal logic:
        ADX > 25 AND +DI > -DI AND MACD > 0 → +min(1.0, ADX/50)
        ADX > 25 AND -DI > +DI AND MACD < 0 → -min(1.0, ADX/50)
        ADX < 25 → 0.0
        DI and MACD disagree → 0.0
    """

    name = "long_I_4h_v12"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 46  # max(adx_period, macd_slow) + 20

    # Optimizable parameters
    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
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
        """Precompute ADX/DI and MACD arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._adx, self._plus_di, self._minus_di = adx_with_di(
            self._highs, self._lows, self._closes, period=self.adx_period,
        )
        self._macd_line, _, _ = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from ADX strength filtered by DI + MACD agreement."""
        adx_val = self._adx[bar_index]
        plus_di = self._plus_di[bar_index]
        minus_di = self._minus_di[bar_index]
        macd_val = self._macd_line[bar_index]

        if np.isnan(adx_val) or np.isnan(plus_di) or np.isnan(macd_val):
            return 0.0

        if adx_val < 25.0:
            return 0.0

        strength = min(1.0, adx_val / 50.0)

        if plus_di > minus_di and macd_val > 0.0:
            return strength
        if minus_di > plus_di and macd_val < 0.0:
            return -strength

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "adx", "params": {"period": self.adx_period}},
            {"name": "macd", "params": {"fast": self.macd_fast, "slow": self.macd_slow}},
        ]
