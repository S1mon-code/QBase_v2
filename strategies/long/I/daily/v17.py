"""MildTrendLongIDailyV17 — KAMA Slope + OI Flow Direction.

Economic logic: KAMA adapts its smoothing to market efficiency — it
accelerates in trending markets and slows in choppy ones, making its
slope a noise-filtered trend signal. OI flow confirmation ensures that
the directional price move is backed by position-building activity,
distinguishing genuine trends from short-covering or technical bounces.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV17(TrendingStrategy):
    """KAMA slope + OI flow direction (Medium Horizon).

    Signal logic:
        - KAMA slope > 0 AND OI flow > OI signal  →  positive signal
        - KAMA slope < 0 AND OI flow < OI signal  →  negative signal
        - No OI confirmation  →  0.0

    Signal strength is normalised slope (% per bar), capped at 1.0.

    Attributes:
        kama_period:     Period for KAMA efficiency ratio window.
        oi_period:       Period for OI flow signal smoothing.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v17"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 59

    kama_period: int = 20
    oi_period: int = 14
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
        """Precompute KAMA and OI flow arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oi_period
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on KAMA slope and OI flow."""
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1] if bar_index > 0 else k
        oi_fl = self._oi_flow[bar_index]
        oi_sig = self._oi_signal[bar_index]

        if np.isnan(k) or np.isnan(k_prev) or np.isnan(oi_fl) or np.isnan(oi_sig):
            return 0.0

        kama_slope = k - k_prev
        kama_sign = 1.0 if kama_slope > 0 else -1.0 if kama_slope < 0 else 0.0

        if kama_sign == 0.0:
            return 0.0

        oi_confirms = (kama_sign > 0 and oi_fl > oi_sig) or (kama_sign < 0 and oi_fl < oi_sig)

        norm_slope = abs(kama_slope) / (k + 1e-10) * 100
        strength = min(1.0, norm_slope * 10)

        return kama_sign * strength if oi_confirms else 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "kama", "params": {"period": self.kama_period}},
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40")
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
