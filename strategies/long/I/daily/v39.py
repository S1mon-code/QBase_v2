"""MildTrendLongIDailyV39 — Donchian Breakout + Force Index Surge.

Economic logic: Donchian channels define the recent price range; a close
beyond the channel boundary signals a structural breakout from equilibrium.
The Force Index (price change × volume) measures the power behind each move.
A breakout confirmed by a positive Force Index surge indicates genuine buying
pressure and is more likely to initiate a sustained trend. Inside-channel
positions provide a half-strength anticipatory signal when Force Index aligns
with price position relative to the midpoint.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV39(TrendingStrategy):
    """Donchian channel breakout filtered by Force Index direction.

    Signal logic:
        - Close > upper channel AND Force Index > 0: +1.0 (confirmed breakout up)
        - Close < lower channel AND Force Index < 0: -1.0 (confirmed breakdown)
        - Close > midline AND Force Index > 0:       +0.5 (pre-breakout buildup)
        - Close < midline AND Force Index < 0:       -0.5 (pre-breakdown pressure)
        - Otherwise:                                   0.0

    Attributes:
        dc_period:       Donchian channel lookback period.
        fi_period:       Force Index EMA smoothing period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v39"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 73  # dc_period(40) + fi_period(13) + 20

    dc_period: int = 40
    fi_period: int = 13
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
        """Precompute Donchian channel bands and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_middle = donchian(
            self._highs,
            self._lows,
            period=self.dc_period,
        )
        self._fi = force_index(
            self._closes,
            self._volumes,
            period=self.fi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return breakout/pre-breakout signal gated by Force Index direction."""
        upper = self._dc_upper[bar_index]
        lower = self._dc_lower[bar_index]
        middle = self._dc_middle[bar_index]
        close = self._closes[bar_index]
        fi = self._fi[bar_index]

        if np.isnan(upper) or np.isnan(lower) or np.isnan(middle) or np.isnan(fi):
            return 0.0

        fi_pos = fi > 0
        fi_neg = fi < 0
        above_mid = close > middle

        if close > upper and fi_pos:
            return 1.0
        if close < lower and fi_neg:
            return -1.0
        if above_mid and fi_pos:
            return 0.5
        if not above_mid and fi_neg:
            return -0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "donchian", "params": {"period": self.dc_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"DC Mid({self.dc_period})", datetimes, self._dc_middle, color="#ffab40"),
            self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, style="dash", color="#26a69a")
        ]
        subplots = [
            self._make_subplot(
                f"Force Index({self.fi_period})",
                [self._make_subplot_trace("Force Index", datetimes, self._fi, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
