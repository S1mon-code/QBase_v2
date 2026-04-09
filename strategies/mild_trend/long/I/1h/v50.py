"""MildTrendLongI1hV50 — Dual RSI(5/14) + OI Flow.

Economic logic: A fast RSI(5) captures short-term momentum shifts while a slow
RSI(14) filters for the intermediate trend. When both RSIs agree on direction
(both above or below 50), the trend signal is strong. OI Flow confirmation
ensures that open interest is flowing in the same direction — new money entering
the market supports the trend. Without OI confirmation, the signal is dampened.
This triple-layer filter reduces false signals in iron ore's noisy 1h data.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV50(TrendingStrategy):
    """Dual RSI agreement confirmed by OI Flow.

    Signal logic:
        - RSI_fast > 50 AND RSI_slow > 50 AND flow > signal:
            +min(1.0, (rsi_fast - 50) / 50)
        - RSI_fast < 50 AND RSI_slow < 50 AND flow < signal:
            -min(1.0, (50 - rsi_fast) / 50)
        - RSIs agree but no OI confirm: signal * 0.4
        - RSIs disagree: 0.0
    """

    name = "mild_trend_long_I_1h_v50"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 48  # rsi_slow_period(14) + oi_period(14) + 20

    rsi_fast_period: int = 5
    rsi_slow_period: int = 14
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
        """Precompute dual RSI and OI Flow arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._rsi_fast = rsi(self._closes, period=self.rsi_fast_period)
        self._rsi_slow = rsi(self._closes, period=self.rsi_slow_period)
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on dual RSI and OI Flow."""
        rf = self._rsi_fast[bar_index]
        rs = self._rsi_slow[bar_index]
        flow_val = self._oi_flow[bar_index]
        sig_val = self._oi_signal[bar_index]

        if (
            np.isnan(rf)
            or np.isnan(rs)
            or np.isnan(flow_val)
            or np.isnan(sig_val)
        ):
            return 0.0

        # Both RSIs bullish
        if rf > 50 and rs > 50:
            strength = min(1.0, (rf - 50.0) / 50.0)
            if flow_val > sig_val:
                return strength
            return strength * 0.4

        # Both RSIs bearish
        if rf < 50 and rs < 50:
            strength = min(1.0, (50.0 - rf) / 50.0)
            if flow_val < sig_val:
                return -strength
            return -strength * 0.4

        # RSIs disagree
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "rsi_fast", "params": {"period": self.rsi_fast_period}},
            {"name": "rsi_slow", "params": {"period": self.rsi_slow_period}},
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"RSI({self.rsi_fast_period}/{self.rsi_slow_period})",
                [
                    self._make_subplot_trace(f"RSI({self.rsi_fast_period})", datetimes, self._rsi_fast, color="#bb86fc"),
                    self._make_subplot_trace(f"RSI({self.rsi_slow_period})", datetimes, self._rsi_slow, color="#4fc3f7"),
                ],
                horizontal_lines=[30, 70], y_range=[0, 100],
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

