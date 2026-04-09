"""MildTrendLongI1hV44 — Vortex Indicator + Volume Momentum.

Economic logic: The Vortex Indicator captures directional movement by comparing
successive highs/lows against each other, producing VI+ and VI- lines. When the
spread between VI+ and VI- widens and volume momentum confirms that participation
is expanding (vol_mom > 1.1), it signals a genuine trending move in iron ore.
Low volume momentum dampens the signal to avoid whipsaws during quiet periods.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.vortex import vortex
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV44(TrendingStrategy):
    """Vortex direction confirmed by volume momentum.

    Signal logic:
        - VI+ > VI- AND vol_mom > 1.1: +(VI+ - VI-)
        - VI- > VI+ AND vol_mom > 1.1: -(VI- - VI+)
        - vol_mom <= 1.1: signal * 0.3
        - Clipped to [-1, 1]
    """

    name = "mild_trend_long_I_1h_v44"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["trend", "volume"]
    warmup: int = 48  # vortex_period(14) + vol_mom_period(14) + 20

    vortex_period: int = 14
    vol_mom_period: int = 14
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
        """Precompute Vortex and Volume Momentum arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period,
        )
        self._vol_mom = volume_momentum(self._volumes, period=self.vol_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on Vortex and Volume Momentum."""
        vip = self._vi_plus[bar_index]
        vim = self._vi_minus[bar_index]
        vm = self._vol_mom[bar_index]

        if np.isnan(vip) or np.isnan(vim) or np.isnan(vm):
            return 0.0

        if vip > vim:
            raw = vip - vim
        elif vim > vip:
            raw = -(vim - vip)
        else:
            return 0.0

        if vm <= 1.1:
            raw *= 0.3

        return float(np.clip(raw, -1.0, 1.0))

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "vortex", "params": {"period": self.vortex_period}},
            {"name": "volume_momentum", "params": {"period": self.vol_mom_period}},
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

