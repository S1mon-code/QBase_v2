"""MildTrendLongIDailyV14 — ADX Trend Strength + DI Direction + CMF Confirmation.

Economic logic: ADX measures the strength of any trend without regard to direction.
The directional indicators (+DI vs -DI) determine whether the trend is up or down.
Chaikin Money Flow filters out false signals by requiring volume to flow in the
same direction as price. Together, they select only high-conviction directional
trends backed by volume participation.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV14(TrendingStrategy):
    """ADX trend strength with DI direction and CMF volume filter.

    Signal logic:
        - ADX < 20: no signal (0.0)
        - ADX 20–threshold: half-strength DI signal if CMF agrees
        - ADX > threshold: full-strength DI signal if CMF agrees
        - CMF disagreement with DI direction: 0.0 (blocked)

    Attributes:
        adx_period:      ADX and DI calculation period.
        cmf_period:      Chaikin Money Flow lookback period.
        adx_threshold:   ADX level above which full signal is issued.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v14"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 68  # adx_period*2(28) + cmf_period(20) + 20

    adx_period: int = 14
    cmf_period: int = 20
    adx_threshold: float = 25.0
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
        """Precompute ADX, +DI, -DI, and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._adx, self._plus_di, self._minus_di = adx_with_di(
            self._highs,
            self._lows,
            self._closes,
            period=self.adx_period,
        )
        self._cmf = cmf(
            self._highs,
            self._lows,
            self._closes,
            self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return ADX/DI directional signal gated by CMF volume confirmation."""
        adx_val = self._adx[bar_index]
        pdi = self._plus_di[bar_index]
        mdi = self._minus_di[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(adx_val) or np.isnan(pdi) or np.isnan(mdi) or np.isnan(cmf_val):
            return 0.0

        if adx_val < 20.0:
            return 0.0

        di_sign = 1.0 if pdi > mdi else -1.0
        cmf_ok = (di_sign > 0 and cmf_val > 0) or (di_sign < 0 and cmf_val < 0)

        if not cmf_ok:
            return 0.0

        strength = 1.0 if adx_val > self.adx_threshold else 0.5
        return di_sign * strength

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "adx_with_di",
                "params": {
                    "period": self.adx_period,
                    "threshold": self.adx_threshold,
                },
            },
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
                f"ADX({self.adx_period})",
                [
                    self._make_subplot_trace("ADX", datetimes, self._adx, color="#bb86fc"),
                    self._make_subplot_trace("+DI", datetimes, self._plus_di, color="#26a69a"),
                    self._make_subplot_trace("-DI", datetimes, self._minus_di, color="#ef5350"),
                ],
            )
        ]
        return {"overlays": [], "subplots": subplots}
