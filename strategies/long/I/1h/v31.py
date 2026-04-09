"""MildTrendLongI1hV31 — RSI Pullback + EMA Trend Filter.

Economic logic: Retail traders chase momentum at extremes and get stopped out
on mean-reversion wiggles.  We enter only when RSI confirms a pullback has
ended (crosses 50) *and* the broader EMA trend is intact, capturing the
continuation move that flushes late counter-trend traders.
"""
from __future__ import annotations

import numpy as np

from indicators._utils import _ema
from indicators.momentum.rsi import rsi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV31(TrendingStrategy):
    """RSI(14) cross-50 in the direction of EMA(20) trend.

    Signal logic:
        Price > EMA AND RSI crosses above 50 → long.
        Price < EMA AND RSI crosses below 50 → short.
        Strength scaled by distance of RSI from 50.
    """

    name = "long_I_1h_v31"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 54  # ema_period + rsi_period + 20

    # Optimizable parameters
    rsi_period: int = 14
    ema_period: int = 20
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
        """Precompute RSI and EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._rsi = rsi(self._closes, period=self.rsi_period)
        self._ema = _ema(self._closes, self.ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal based on RSI cross-50 with EMA trend filter."""
        rsi_val = self._rsi[bar_index]
        rsi_prev = self._rsi[bar_index - 1]
        ema_val = self._ema[bar_index]
        close = self._closes[bar_index]

        if np.isnan(rsi_val) or np.isnan(rsi_prev) or np.isnan(ema_val):
            return 0.0

        # Strength: scale by distance from 50 → 0..1
        distance = abs(rsi_val - 50.0) / 50.0
        strength = min(1.0, distance)

        # Long: price above EMA and RSI crosses above 50
        if close > ema_val and rsi_val > 50.0 and rsi_prev <= 50.0:
            return strength
        # Short: price below EMA and RSI crosses below 50
        if close < ema_val and rsi_val < 50.0 and rsi_prev >= 50.0:
            return -strength

        # Sustain weaker signal while conditions hold
        if close > ema_val and rsi_val > 50.0:
            return strength * 0.5
        if close < ema_val and rsi_val < 50.0:
            return -strength * 0.5

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "rsi", "params": {"period": self.rsi_period}},
            {"name": "ema", "params": {"period": self.ema_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"EMA({self.ema_period})", datetimes, self._ema, color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"RSI({self.rsi_period})",
                [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc")],
                horizontal_lines=[30, 70], y_range=[0, 100],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}

