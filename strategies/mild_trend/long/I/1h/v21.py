"""MildTrendLongI1hV21 — Vortex Indicator + Volume Momentum.

Economic logic: The Vortex Indicator captures positive and negative trend
movement via true range ratios. When VI+ exceeds VI-, a bullish trend is
confirmed. Volume Momentum (volume vs its SMA, smoothed by EMA) validates
that participation is increasing alongside the trend. Both confirming
together signals institutional conviction behind the move; VI+ alone yields
a tentative long. Lookbacks suited for 1H iron ore capturing multi-day swings.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.vortex import vortex
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV21(TrendingStrategy):
    """Vortex bullish crossover with volume momentum confirmation.

    Signal logic:
        VI+ > VI- AND vol_mom > 1.0 -> strong signal (0.7-1.0)
        VI+ > VI- only              -> weak signal (0.3-0.5)
        else                        -> 0.0
    """

    name = "mild_trend_long_I_1h_v21"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50  # vortex_period(30) + vol_mom needs 2*period, + buffer

    # Optimizable parameters (<=5 including chandelier_mult)
    vortex_period: int = 24
    vol_mom_period: int = 20
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
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period
        )
        self._vol_mom = volume_momentum(self._volumes, period=self.vol_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        vi_p = self._vi_plus[bar_index]
        vi_m = self._vi_minus[bar_index]
        vm = self._vol_mom[bar_index]

        if np.isnan(vi_p) or np.isnan(vi_m):
            return 0.0

        if vi_p <= vi_m:
            return 0.0

        # Vortex spread as trend strength measure
        vortex_spread = vi_p - vi_m
        base_signal = min(1.0, 0.3 + vortex_spread * 2.0)

        if not np.isnan(vm) and vm > 1.0:
            # Volume momentum confirms — boost signal
            vol_boost = min(0.3, (vm - 1.0) * 0.5)
            return min(1.0, base_signal + vol_boost)

        return min(0.5, base_signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "VI+", "array": self._vi_plus, "panel": "Vortex"},
            {"name": "VI-", "array": self._vi_minus, "panel": "Vortex"},
            {"name": "Volume Momentum", "array": self._vol_mom},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"Vortex({self.vortex_period})",
                [
                    self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#26a69a"),
                    self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                ],
            ),
            self._make_subplot(
                f"Vol Mom({self.vol_mom_period})",
                [self._make_subplot_trace("Vol Mom", datetimes, self._vol_mom, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}

