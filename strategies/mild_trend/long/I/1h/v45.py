"""MildTrendLongI1hV45 — HMA Slope + Force Index.

Economic logic: Hull Moving Average reacts faster than standard MAs while
remaining smooth. Its slope direction indicates momentum. The Force Index
(price change x volume) confirms that the move is backed by genuine
participation. ATR normalizes the HMA slope so that signal strength is
comparable across different volatility regimes in the iron-ore market.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV45(TrendingStrategy):
    """HMA slope direction confirmed by Force Index, ATR-normalized.

    Signal logic:
        - HMA rising AND FI > 0: +min(1.0, abs(hma_slope) / atr * 5)
        - HMA falling AND FI < 0: -min(1.0, abs(hma_slope) / atr * 5)
        - Disagreement: 0.0
    """

    name = "mild_trend_long_I_1h_v45"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["trend", "volume"]
    warmup: int = 43  # hma_period(10) + fi_period(13) + 20

    hma_period: int = 10
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
        """Precompute HMA, Force Index, and ATR arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)
        self._atr = atr(self._highs, self._lows, self._closes, period=14)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on HMA slope and Force Index."""
        if bar_index < 1:
            return 0.0

        hma_cur = self._hma[bar_index]
        hma_prev = self._hma[bar_index - 1]
        fi_val = self._fi[bar_index]
        atr_val = self._atr[bar_index]

        if (
            np.isnan(hma_cur)
            or np.isnan(hma_prev)
            or np.isnan(fi_val)
            or np.isnan(atr_val)
            or atr_val <= 0
        ):
            return 0.0

        hma_slope = hma_cur - hma_prev
        hma_rising = hma_slope > 0
        hma_falling = hma_slope < 0

        strength = min(1.0, abs(hma_slope) / atr_val * 5.0)

        if hma_rising and fi_val > 0:
            return strength
        if hma_falling and fi_val < 0:
            return -strength
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "hma", "params": {"period": self.hma_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
            {"name": "atr", "params": {"period": 14}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"Force Index({self.fi_period})",
                [self._make_subplot_trace("Force Index", datetimes, self._fi, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}

