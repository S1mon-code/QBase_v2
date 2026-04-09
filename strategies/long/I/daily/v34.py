"""MildTrendLongIDailyV34 — ADX Trend Strength Filter + MACD Direction Alignment.

Economic logic: ADX above 25 confirms that price is in a genuine trending
regime rather than ranging. Within that confirmed regime, MACD line sign
(fast EMA minus slow EMA) provides the direction, while the DI crossover
supplies a second independent vote. Requiring both to agree eliminates
mixed signals and keeps the strategy silent in ambiguous conditions.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.adx import adx_with_di
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV34(TrendingStrategy):
    """ADX-filtered trend with MACD line and DI direction alignment.

    Signal logic:
        - ADX > 25 AND +DI > -DI AND MACD line > 0: +1.0
        - ADX > 25 AND -DI > +DI AND MACD line < 0: -1.0
        - ADX > 25 but DI and MACD disagree: 0.0
        - ADX <= 25: 0.0 (no trending regime)

    Attributes:
        adx_period:      ADX / DI smoothing period.
        macd_fast:       MACD fast EMA period.
        macd_slow:       MACD slow EMA period.
        signal_period:   MACD signal line period (unused in signal, for completeness).
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v34"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 74  # adx_period*2(28) + macd_slow(26) + 20

    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    signal_period: int = 9
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
        """Precompute ADX/DI and MACD line arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._adx, self._plus_di, self._minus_di = adx_with_di(
            self._highs,
            self._lows,
            self._closes,
            period=self.adx_period,
        )
        self._macd_line, _, _ = macd(
            self._closes,
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.signal_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on ADX regime and MACD/DI alignment."""
        adx_val = self._adx[bar_index]
        pdi = self._plus_di[bar_index]
        mdi = self._minus_di[bar_index]
        macd_val = self._macd_line[bar_index]

        if np.isnan(adx_val) or np.isnan(pdi) or np.isnan(mdi) or np.isnan(macd_val):
            return 0.0

        if adx_val < 25:
            return 0.0

        di_up = pdi > mdi
        macd_up = macd_val > 0

        if di_up and macd_up:
            return 1.0
        if not di_up and not macd_up:
            return -1.0
        return 0.0  # mixed signals

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "adx_with_di", "params": {"period": self.adx_period}},
            {
                "name": "macd",
                "params": {
                    "fast": self.macd_fast,
                    "slow": self.macd_slow,
                    "signal": self.signal_period,
                },
            },
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"MACD({self.macd_fast},{self.macd_slow})",
                [self._make_subplot_trace("MACD Line", datetimes, self._macd_line, color="#bb86fc")],
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
