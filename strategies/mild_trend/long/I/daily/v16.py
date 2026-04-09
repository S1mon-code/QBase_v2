"""MildTrendLongIDailyV16 — Vortex Cross + CMF Direction.

Economic logic: The Vortex Indicator captures directional trend strength
by measuring upward and downward movement relative to true range.
Confirmed by Chaikin Money Flow, which validates that capital is flowing
in the direction of the vortex cross, reducing false signals in choppy
markets.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.vortex import vortex
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV16(TrendingStrategy):
    """Vortex cross + CMF direction confirmation (Medium Horizon).

    Signal logic:
        - VI+ > VI- AND CMF > 0  →  full positive signal
        - VI- > VI+ AND CMF < 0  →  full negative signal
        - Cross disagrees with CMF  →  0.3 * vortex direction

    Attributes:
        vortex_period:   Period for Vortex Indicator calculation.
        cmf_period:      Period for Chaikin Money Flow.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v16"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 54

    vortex_period: int = 14
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
        """Precompute Vortex and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes, period=self.cmf_period
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on Vortex cross and CMF confirmation."""
        vi_p = self._vi_plus[bar_index]
        vi_m = self._vi_minus[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(vi_p) or np.isnan(vi_m) or np.isnan(cmf_val):
            return 0.0

        vortex_sign = 1.0 if vi_p > vi_m else -1.0
        cmf_ok = (vortex_sign > 0 and cmf_val > 0) or (vortex_sign < 0 and cmf_val < 0)

        vi_diff = abs(vi_p - vi_m)
        strength = min(1.0, vi_diff / 0.1)

        return vortex_sign * strength if cmf_ok else vortex_sign * 0.3

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "vortex", "params": {"period": self.vortex_period}},
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                f"Vortex({self.vortex_period})",
                [
                    self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#26a69a"),
                    self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                ],
            )
        ]
        return {"overlays": [], "subplots": subplots}
