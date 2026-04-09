"""MildTrendLongI1hV15 — ADX Trend + Aroon Direction.

Economic logic: ADX measures trend strength regardless of direction, while
Aroon identifies whether bulls or bears are dominant by tracking time since
recent highs vs lows. When ADX confirms a strong trend (>25) and Aroon shows
bullish dominance (aroon_up > aroon_down), the strategy enters with conviction
scaled by ADX strength. This dual-filter avoids choppy, directionless markets.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx
from indicators.trend.aroon import aroon
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV15(TrendingStrategy):
    """ADX > 25 with Aroon bullish dominance.

    Signal logic:
        adx > 25 AND aroon_up > aroon_down
            -> signal = min(1.0, (adx - 25) / 50 + 0.3)
        else -> 0.0
    """

    name = "long_I_1h_v15"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 50  # max(adx_period, aroon_period)(30) + buffer(20)

    # Optimizable parameters
    adx_period: int = 20
    aroon_period: int = 30
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
        """Precompute ADX and Aroon arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._adx = adx(self._highs, self._lows, self._closes, period=self.adx_period)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal based on ADX trend strength with Aroon directional filter."""
        adx_val = self._adx[bar_index]
        aroon_up = self._aroon_up[bar_index]
        aroon_down = self._aroon_down[bar_index]

        if np.isnan(adx_val) or np.isnan(aroon_up) or np.isnan(aroon_down):
            return 0.0

        if adx_val > 25.0 and aroon_up > aroon_down:
            return min(1.0, (adx_val - 25.0) / 50.0 + 0.3)

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for auto-generated panels."""
        return [
            {"name": "ADX", "array": self._adx},
            {"name": "Aroon Up", "array": self._aroon_up, "panel": "Aroon"},
            {"name": "Aroon Down", "array": self._aroon_down, "panel": "Aroon"},
            {"name": "Aroon Osc", "array": self._aroon_osc, "panel": "Aroon", "style": "area"},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"ADX({self.adx_period})",
                [self._make_subplot_trace("ADX", datetimes, self._adx, color="#bb86fc")],
            ),
            self._make_subplot(
                f"Aroon({self.aroon_period})",
                [
                    self._make_subplot_trace("Aroon Up", datetimes, self._aroon_up, color="#26a69a"),
                    self._make_subplot_trace("Aroon Down", datetimes, self._aroon_down, color="#ef5350"),
                ],
                y_range=[0, 100],
            )
        ]
        return {"overlays": [], "subplots": subplots}
