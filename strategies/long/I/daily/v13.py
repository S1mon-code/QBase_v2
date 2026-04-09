"""MildTrendLongIDailyV13 — SuperTrend + Volume Momentum Filter.

Economic logic: SuperTrend provides an ATR-based adaptive trend channel that
flips direction on genuine breakouts. Volume momentum (ratio of current volume
to its smoothed average) confirms that the trend is backed by institutional
participation. Strong volume confirmation yields a full signal; weak volume yields
a partial signal; below-average volume suppresses the signal.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV13(TrendingStrategy):
    """SuperTrend direction scaled by volume momentum strength.

    Signal logic:
        - direction +1 (bullish) / -1 (bearish) from SuperTrend
        - volume_mom > 1.2: full signal (strength = 1.0)
        - volume_mom 1.0–1.2: moderate signal (strength = 0.6)
        - volume_mom < 1.0: weak signal (strength = 0.3)

    Attributes:
        st_period:       SuperTrend ATR period.
        st_mult:         SuperTrend ATR multiplier.
        vol_mom_period:  Volume momentum lookback period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v13"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 74  # st_period(14) + vol_mom_period*2(40) + 20

    st_period: int = 14
    st_mult: float = 3.0
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
        """Precompute SuperTrend and volume momentum arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        _, self._st_direction = supertrend(
            self._highs,
            self._lows,
            self._closes,
            period=self.st_period,
            multiplier=self.st_mult,
        )
        self._vol_mom = volume_momentum(self._volumes, period=self.vol_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return SuperTrend-directional signal scaled by volume momentum."""
        direction = self._st_direction[bar_index]
        vmom = self._vol_mom[bar_index]

        if np.isnan(direction) or np.isnan(vmom):
            return 0.0

        base = float(direction)  # +1.0 or -1.0
        strength = 1.0 if vmom > 1.2 else 0.6 if vmom > 1.0 else 0.3
        return base * strength

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "supertrend",
                "params": {
                    "period": self.st_period,
                    "multiplier": self.st_mult,
                },
            },
            {
                "name": "volume_momentum",
                "params": {"period": self.vol_mom_period},
            },
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"Vol Mom({self.vol_mom_period})",
                [self._make_subplot_trace("Vol Mom", datetimes, self._vol_mom, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                f"ST Dir({self.st_period})",
                [self._make_subplot_trace("Direction", datetimes, self._st_direction, style="step", color="#4fc3f7")],
            )
        ]
        return {"overlays": [], "subplots": subplots}
