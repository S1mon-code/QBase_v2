"""MildTrendLongIDailyV26 — KAMA Slope + Force Index Confirmation.

Economic logic: KAMA (Kaufman Adaptive Moving Average) adapts its speed to
market efficiency, smoothing noise while tracking genuine trends. Its slope
reveals whether price is trending. Force Index (Elder) combines price change
direction, magnitude, and volume — positive values confirm buying pressure,
negative values confirm selling pressure. When both agree, a high-conviction
trend signal is produced.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV26(TrendingStrategy):
    """KAMA slope confirmed by Force Index.

    Signal logic:
        - KAMA slope > 0 AND Force Index > 0: long (strength = normalised slope)
        - KAMA slope < 0 AND Force Index < 0: short (strength = normalised slope)
        - Either flat (slope == 0) or indicators disagree: 0.0

    Attributes:
        kama_period:     KAMA lookback period.
        fi_period:       Force Index EMA smoothing period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v26"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 53  # kama_period(20) + fi_period(13) + 20

    kama_period: int = 20
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
        """Precompute KAMA and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on KAMA slope and Force Index agreement."""
        if bar_index < 1:
            return 0.0

        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1]
        fi = self._fi[bar_index]

        if np.isnan(k) or np.isnan(k_prev) or np.isnan(fi):
            return 0.0

        slope = k - k_prev
        kama_sign = 1.0 if slope > 0 else -1.0 if slope < 0 else 0.0

        if kama_sign == 0.0:
            return 0.0

        fi_ok = (kama_sign > 0 and fi > 0) or (kama_sign < 0 and fi < 0)
        if not fi_ok:
            return 0.0

        norm_slope = min(1.0, abs(slope) / (k + 1e-10) * 200)
        return kama_sign * max(0.3, norm_slope)

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "kama", "params": {"period": self.kama_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"Force Index({self.fi_period})",
                [self._make_subplot_trace("Force Index", datetimes, self._fi, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
