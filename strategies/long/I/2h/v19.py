"""MildTrendLongI2hV19 — Donchian Channel Breakout + OI Flow Confirmation.

Economic logic: Donchian channel breakouts signal that price has exceeded its
recent range — a classic trend-following entry. OI Flow combines open interest
changes with volume to detect whether institutional positioning supports the
breakout. A breakout confirmed by rising OI flow is far more likely to sustain
than one driven by retail activity alone.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV19(TrendingStrategy):
    """Donchian breakout with OI flow institutional confirmation.

    Signal logic:
        - Close > upper AND flow > signal → +1.0
        - Close < lower AND flow < signal → -1.0
        - Close > mid AND flow > signal → +0.5
        - Close < mid AND flow < signal → -0.5
        - Else → 0.0

    Attributes:
        donch_period:    Donchian channel lookback period.
        oi_period:       OI flow calculation period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_2h_v19"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 54  # donch_period + oi_period + 20

    donch_period: int = 20
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
        """Precompute Donchian and OI Flow arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._donch_upper, self._donch_mid, self._donch_lower = donchian(
            self._highs, self._lows, period=self.donch_period,
        )
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return Donchian breakout signal confirmed by OI flow."""
        close_val = self._closes[bar_index]
        upper = self._donch_upper[bar_index]
        mid = self._donch_mid[bar_index]
        lower = self._donch_lower[bar_index]
        flow_val = self._oi_flow[bar_index]
        flow_sig = self._oi_signal[bar_index]

        if (
            np.isnan(close_val) or np.isnan(upper) or np.isnan(mid)
            or np.isnan(lower) or np.isnan(flow_val) or np.isnan(flow_sig)
        ):
            return 0.0

        flow_bullish = flow_val > flow_sig
        flow_bearish = flow_val < flow_sig

        if close_val > upper and flow_bullish:
            return 1.0
        if close_val < lower and flow_bearish:
            return -1.0
        if close_val > mid and flow_bullish:
            return 0.5
        if close_val < mid and flow_bearish:
            return -0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "donchian", "params": {"period": self.donch_period}},
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]
