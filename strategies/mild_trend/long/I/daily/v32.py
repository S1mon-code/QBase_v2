"""MildTrendLongIDailyV32 — Triple EMA Slope Agreement + OI Flow Confirmation.

Economic logic: When three EMAs spanning different horizons (20, 40, 80 bars)
simultaneously slope in the same direction, the trend is coherent across
time-frames — a high-quality trending environment. OI flow confirmation
ensures that open interest is accumulating in the direction of the trend,
signalling genuine position-building rather than noise-driven price action.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV32(TrendingStrategy):
    """Three-EMA slope agreement scaled by OI flow direction.

    Signal logic:
        - All three EMA slopes positive AND OI flow > OI signal: +1.0
        - All three EMA slopes negative AND OI flow < OI signal: -1.0
        - All three slopes positive (no OI confirm): +0.7
        - All three slopes negative (no OI confirm): -0.7
        - Two slopes positive AND OI agrees: +0.5
        - Two slopes negative AND OI agrees: -0.5
        - Otherwise: 0.0

    Attributes:
        p1:              EMA fast period.
        p2:              EMA medium period.
        p3:              EMA slow period.
        oi_period:       OI flow EMA signal period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v32"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 120  # p3(80) + oi_period(20) + 20

    p1: int = 20
    p2: int = 40
    p3: int = 80
    oi_period: int = 20
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
        """Precompute three EMA arrays and OI flow arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema1 = ema(self._closes, period=self.p1)
        self._ema2 = ema(self._closes, period=self.p2)
        self._ema3 = ema(self._closes, period=self.p3)
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes,
            self._oi,
            self._volumes,
            period=self.oi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on EMA slope agreement and OI flow."""
        i = bar_index
        if i < 1:
            return 0.0

        s1 = self._ema1[i] - self._ema1[i - 1]
        s2 = self._ema2[i] - self._ema2[i - 1]
        s3 = self._ema3[i] - self._ema3[i - 1]
        oi_fl = self._oi_flow[i]
        oi_sig = self._oi_signal[i]

        if any(np.isnan(v) for v in [s1, s2, s3, oi_fl, oi_sig]):
            return 0.0

        bull_slopes = sum(s > 0 for s in [s1, s2, s3])
        bear_slopes = sum(s < 0 for s in [s1, s2, s3])
        oi_up = oi_fl > oi_sig
        oi_dn = oi_fl < oi_sig

        if bull_slopes == 3 and oi_up:
            return 1.0
        if bear_slopes == 3 and oi_dn:
            return -1.0
        if bull_slopes == 3:
            return 0.7
        if bear_slopes == 3:
            return -0.7
        if bull_slopes == 2 and oi_up:
            return 0.5
        if bear_slopes == 2 and oi_dn:
            return -0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "ema", "params": {"period": self.p1}},
            {"name": "ema", "params": {"period": self.p2}},
            {"name": "ema", "params": {"period": self.p3}},
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"EMA({self.p1})", datetimes, self._ema1, color="#ffab40"),
            self._make_overlay(f"EMA({self.p2})", datetimes, self._ema2, color="#ab47bc"),
            self._make_overlay(f"EMA({self.p3})", datetimes, self._ema3, color="#4fc3f7")
        ]
        subplots = [
            self._make_subplot(
                f"OI Flow({self.oi_period})",
                [
                    self._make_subplot_trace("OI Flow", datetimes, self._oi_flow, color="#bb86fc"),
                    self._make_subplot_trace("OI Signal", datetimes, self._oi_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
