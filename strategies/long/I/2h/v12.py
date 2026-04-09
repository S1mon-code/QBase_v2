"""MildTrendLongI2hV12 — ADX Trend Strength + Stochastic K/D Confirmation.

Economic logic: ADX quantifies trend strength irrespective of direction, while
the directional indicators (+DI / -DI) reveal trend polarity. Stochastic K/D
adds momentum timing — K above D confirms bullish momentum, K below D confirms
bearish. The combination selects strong, directionally-confirmed trends with
proper momentum alignment.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.momentum.stochastic import stochastic
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV12(TrendingStrategy):
    """ADX trend strength with Stochastic K/D momentum confirmation.

    Signal logic:
        - ADX > 25 AND +DI > -DI AND K > D → +min(1.0, adx/50)
        - ADX > 25 AND -DI > +DI AND K < D → -min(1.0, adx/50)
        - ADX 15–25 → signal * 0.3
        - ADX < 15 → 0.0

    Attributes:
        adx_period:      ADX and DI calculation period.
        k_period:        Stochastic %K period.
        d_period:        Stochastic %D smoothing period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_2h_v12"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 48  # adx_period + k_period + 20

    adx_period: int = 14
    k_period: int = 14
    d_period: int = 3
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
        """Precompute ADX, DI, and Stochastic arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._adx, self._plus_di, self._minus_di = adx_with_di(
            self._highs, self._lows, self._closes, period=self.adx_period,
        )
        self._k, self._d = stochastic(
            self._highs, self._lows, self._closes,
            k_period=self.k_period, d_period=self.d_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return ADX/DI directional signal confirmed by Stochastic K/D."""
        adx_val = self._adx[bar_index]
        pdi = self._plus_di[bar_index]
        mdi = self._minus_di[bar_index]
        k_val = self._k[bar_index]
        d_val = self._d[bar_index]

        if (
            np.isnan(adx_val) or np.isnan(pdi) or np.isnan(mdi)
            or np.isnan(k_val) or np.isnan(d_val)
        ):
            return 0.0

        if adx_val < 15.0:
            return 0.0

        # Determine direction and stochastic confirmation
        if pdi > mdi and k_val > d_val:
            raw = min(1.0, adx_val / 50.0)
        elif mdi > pdi and k_val < d_val:
            raw = -min(1.0, adx_val / 50.0)
        else:
            return 0.0

        # Scale by ADX regime
        if adx_val < 25.0:
            return raw * 0.3
        return raw

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "adx_with_di", "params": {"period": self.adx_period}},
            {
                "name": "stochastic",
                "params": {"k_period": self.k_period, "d_period": self.d_period},
            },
        ]
