"""MildTrendLongIDailyV31 — MACD Histogram Zero Cross + Volume Momentum Surge.

Economic logic: The MACD histogram crossing zero marks the precise moment the
fast EMA momentum regime flips — a high-conviction entry signal. Pairing this
flip with a volume momentum surge above 1.2 confirms institutional participation,
filtering out low-energy false crosses. Sustained positive histogram with moderate
volume lift yields a partial signal.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV31(TrendingStrategy):
    """MACD histogram zero-cross confirmed by volume momentum surge.

    Signal logic:
        - Histogram crosses negative→positive AND vol_mom > 1.2: +1.0
        - Histogram crosses positive→negative AND vol_mom > 1.2: -1.0
        - Histogram positive (sustained) AND vol_mom > 1.0: +0.5
        - Histogram negative (sustained) AND vol_mom > 1.0: -0.5
        - Otherwise: 0.0

    Attributes:
        fast_period:     MACD fast EMA period.
        slow_period:     MACD slow EMA period.
        signal_period:   MACD signal line period.
        vol_mom_period:  Volume momentum lookback period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v31"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 83  # slow(26) + signal(9) + vol_mom_period*2(28) + 20

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
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
        """Precompute MACD histogram and volume momentum arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        _, _, self._hist = macd(
            self._closes,
            fast=self.fast_period,
            slow=self.slow_period,
            signal=self.signal_period,
        )
        self._vol_mom = volume_momentum(self._volumes, period=self.vol_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on MACD histogram cross and volume momentum."""
        i = bar_index
        hist = self._hist[i]
        hist_prev = self._hist[i - 1] if i > 0 else 0.0
        vmom = self._vol_mom[i]

        if np.isnan(hist) or np.isnan(vmom):
            return 0.0
        if i > 0 and np.isnan(hist_prev):
            hist_prev = 0.0

        crossed_up = hist_prev < 0 and hist > 0
        crossed_dn = hist_prev > 0 and hist < 0

        if crossed_up and vmom > 1.2:
            return 1.0
        if crossed_dn and vmom > 1.2:
            return -1.0
        if hist > 0 and vmom > 1.0:
            return 0.5
        if hist < 0 and vmom > 1.0:
            return -0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "macd",
                "params": {
                    "fast": self.fast_period,
                    "slow": self.slow_period,
                    "signal": self.signal_period,
                },
            },
            {"name": "volume_momentum", "params": {"period": self.vol_mom_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"MACD Hist({self.fast_period},{self.slow_period})",
                [self._make_subplot_trace("Histogram", datetimes, self._hist, style="bar", color_positive="#26a69a", color_negative="#ef5350")],
                zero_line=True,
            ),
            self._make_subplot(
                f"Vol Mom({self.vol_mom_period})",
                [self._make_subplot_trace("Vol Mom", datetimes, self._vol_mom, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
