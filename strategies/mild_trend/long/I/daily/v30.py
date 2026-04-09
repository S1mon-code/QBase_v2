"""MildTrendLongIDailyV30 — Triple EMA Alignment + OBV EMA Trend.

Economic logic: Triple EMA alignment (short > medium > long) is a classical
trend-following filter that requires all timeframes to agree. This eliminates
false signals from partial crossovers. OBV (On-Balance Volume) EMA slope
confirms that volume is accumulating in the trend direction. Full three-way
EMA alignment with OBV confirmation yields the strongest signal; partial
alignment or OBV disagreement reduces conviction proportionally.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV30(TrendingStrategy):
    """Triple EMA alignment confirmed by OBV EMA slope.

    Signal logic:
        - short > medium > long AND OBV_EMA rising: +1.0 (full long)
        - short < medium < long AND OBV_EMA falling: -1.0 (full short)
        - EMA fully aligned but OBV disagrees: ±0.7
        - Two out of three EMA levels aligned with OBV: ±0.4
        - No clear alignment: 0.0

    Attributes:
        short_period:    Short EMA period.
        medium_period:   Medium EMA period.
        long_period:     Long EMA period.
        obv_period:      EMA smoothing period applied to raw OBV.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v30"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 100  # long_period(60) + obv_period(20) + 20

    short_period: int = 10
    medium_period: int = 30
    long_period: int = 60
    obv_period: int = 20
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
        """Precompute short, medium, long EMA and OBV EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_short = ema(self._closes, self.short_period)
        self._ema_medium = ema(self._closes, self.medium_period)
        self._ema_long = ema(self._closes, self.long_period)
        obv_raw = obv(self._closes, self._volumes)
        self._obv_ema = ema(obv_raw, self.obv_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on triple EMA alignment and OBV slope."""
        s = self._ema_short[bar_index]
        m = self._ema_medium[bar_index]
        lo = self._ema_long[bar_index]
        obv_now = self._obv_ema[bar_index]

        if np.isnan(s) or np.isnan(m) or np.isnan(lo) or np.isnan(obv_now):
            return 0.0

        obv_slope = 0.0
        if bar_index > 0 and not np.isnan(self._obv_ema[bar_index - 1]):
            obv_slope = obv_now - self._obv_ema[bar_index - 1]

        bull_full = s > m > lo and obv_slope > 0
        bear_full = s < m < lo and obv_slope < 0

        if bull_full:
            return 1.0
        if bear_full:
            return -1.0
        if s > m and m > lo:
            return 0.7   # EMA fully aligned bull but OBV disagrees
        if s < m and m < lo:
            return -0.7  # EMA fully aligned bear but OBV disagrees
        if (s > m or m > lo) and obv_slope > 0:
            return 0.4
        if (s < m or m < lo) and obv_slope < 0:
            return -0.4
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "ema_triple",
                "params": {
                    "short_period": self.short_period,
                    "medium_period": self.medium_period,
                    "long_period": self.long_period,
                },
            },
            {"name": "obv_ema", "params": {"obv_period": self.obv_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"EMA({self.short_period})", datetimes, self._ema_short, color="#ffab40"),
            self._make_overlay(f"EMA({self.medium_period})", datetimes, self._ema_medium, color="#ab47bc"),
            self._make_overlay(f"EMA({self.long_period})", datetimes, self._ema_long, color="#4fc3f7")
        ]
        subplots = [
            self._make_subplot(
                f"OBV EMA({self.obv_period})",
                [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#bb86fc")],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
