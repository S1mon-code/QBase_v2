"""MildTrendLongI4hV19 — Donchian(30) + RSI Confirmation.

Economic logic: Donchian channel breakouts identify range expansion moments.  RSI
provides momentum confirmation — a breakout above the upper band with RSI > 50
confirms buyers are in control.  RSI magnitude scales the signal strength, giving
stronger conviction to moves with clear momentum backing.  Inside the channel, only
extreme RSI readings generate weak signals.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.donchian import donchian
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV19(TrendingStrategy):
    """Donchian breakout with RSI-scaled signal strength.

    Signal logic:
        Close > upper AND RSI > 50 → +min(1.0, (RSI-50)/50)
        Close < lower AND RSI < 50 → -min(1.0, (50-RSI)/50)
        Inside channel: RSI > 60   → +0.3
        Inside channel: RSI < 40   → -0.3
        Else → 0.0
    """

    name = "mild_trend_long_I_4h_v19"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 64  # donch_period + rsi_period + 20

    # Optimizable parameters
    donch_period: int = 30
    rsi_period: int = 14
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
        """Precompute Donchian channels and RSI."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._donch_upper, _, self._donch_lower = donchian(
            self._highs, self._lows, period=self.donch_period,
        )
        self._rsi = rsi(self._closes, period=self.rsi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from Donchian breakout scaled by RSI magnitude."""
        c = self._closes[bar_index]
        upper = self._donch_upper[bar_index]
        lower = self._donch_lower[bar_index]
        rsi_val = self._rsi[bar_index]

        if np.isnan(c) or np.isnan(upper) or np.isnan(lower) or np.isnan(rsi_val):
            return 0.0

        # Breakout signals with RSI scaling
        if c > upper and rsi_val > 50.0:
            return min(1.0, (rsi_val - 50.0) / 50.0)
        if c < lower and rsi_val < 50.0:
            return -min(1.0, (50.0 - rsi_val) / 50.0)

        # Inside channel — only extreme RSI
        if rsi_val > 60.0:
            return 0.3
        if rsi_val < 40.0:
            return -0.3

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "donchian", "params": {"period": self.donch_period}},
            {"name": "rsi", "params": {"period": self.rsi_period}},
        ]
