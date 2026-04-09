"""MildTrendLongI1hV43 — RSI(2) Extreme + EMA(50) Trend.

Economic logic: Ultra-short RSI(2) reaches extreme oversold/overbought readings
very quickly, identifying sharp pullbacks within established trends. When price
sits above EMA(50) — confirming the underlying uptrend — an RSI(2) < 10 reading
signals a high-probability mean-reversion entry (buy-the-dip) within the trend.
The reverse applies for short entries in downtrends. This is a classic
"pullback in trend" strategy well-suited to iron-ore's tendency for sharp
intra-trend corrections on the 1h chart.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators._utils import _ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV43(TrendingStrategy):
    """RSI(2) extreme pullback entries filtered by EMA(50) trend.

    Signal logic:
        - Close > EMA(50) AND RSI(2) < 10:  +1.0  (extreme dip in uptrend)
        - Close < EMA(50) AND RSI(2) > 90:  -1.0  (extreme bounce in downtrend)
        - Close > EMA(50) AND RSI(2) < 20:  +0.6  (mild dip in uptrend)
        - Close < EMA(50) AND RSI(2) > 80:  -0.6  (mild bounce in downtrend)
        - Else: 0.0
    """

    name = "long_I_1h_v43"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "trend"]
    warmup: int = 70  # ema_period(50) + 20

    rsi_period: int = 2
    ema_period: int = 50
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
        """Precompute RSI(2) and EMA(50) arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._rsi = rsi(self._closes, period=self.rsi_period)
        self._ema = _ema(self._closes, period=self.ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on RSI extreme in EMA trend."""
        rsi_val = self._rsi[bar_index]
        ema_val = self._ema[bar_index]
        close_val = self._closes[bar_index]

        if np.isnan(rsi_val) or np.isnan(ema_val) or np.isnan(close_val):
            return 0.0

        if close_val > ema_val:
            if rsi_val < 10:
                return 1.0
            if rsi_val < 20:
                return 0.6
        elif close_val < ema_val:
            if rsi_val > 90:
                return -1.0
            if rsi_val > 80:
                return -0.6
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

