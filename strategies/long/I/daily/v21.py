"""MildTrendLongIDailyV21 — MACD Line Direction + OI Flow Confirmation.

Economic logic: MACD line (fast EMA minus slow EMA) captures intermediate-term
price momentum direction. OI flow measures whether open interest is accumulating
in alignment with the price move — rising OI above its signal line in a price
uptrend indicates new money entering the long side, confirming the move.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV21(TrendingStrategy):
    """MACD line direction confirmed by OI flow.

    Signal logic:
        - MACD line > 0 AND oi_flow > oi_signal: +1.0 (aligned long)
        - MACD line < 0 AND oi_flow < oi_signal: -1.0 (aligned short)
        - Misaligned: 0.4 * sign(MACD line) (weak directional signal)

    Attributes:
        fast_period:     MACD fast EMA period.
        slow_period:     MACD slow EMA period.
        signal_period:   MACD signal line period.
        oi_period:       OI flow signal EMA period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v21"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 69  # slow_period(26) + signal_period(9) + oi_period(14) + 20

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
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
        """Precompute MACD line, OI flow, and OI flow signal arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._macd_line, _, _ = macd(
            self._closes,
            fast=self.fast_period,
            slow=self.slow_period,
            signal=self.signal_period,
        )
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes,
            self._oi,
            self._volumes,
            period=self.oi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on MACD line and OI flow alignment."""
        macd_val = self._macd_line[bar_index]
        flow_val = self._oi_flow[bar_index]
        sig_val = self._oi_signal[bar_index]

        if np.isnan(macd_val) or np.isnan(flow_val) or np.isnan(sig_val):
            return 0.0

        macd_sign = 1.0 if macd_val > 0 else -1.0 if macd_val < 0 else 0.0
        if macd_sign == 0.0:
            return 0.0

        oi_aligned = (macd_val > 0 and flow_val > sig_val) or (
            macd_val < 0 and flow_val < sig_val
        )
        return macd_sign if oi_aligned else 0.4 * macd_sign

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
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"MACD({self.fast_period},{self.slow_period})",
                [self._make_subplot_trace("MACD Line", datetimes, self._macd_line, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                f"OI Flow({self.oi_period})",
                [
                    self._make_subplot_trace("OI Flow", datetimes, self._oi_flow, color="#bb86fc"),
                    self._make_subplot_trace("OI Signal", datetimes, self._oi_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
