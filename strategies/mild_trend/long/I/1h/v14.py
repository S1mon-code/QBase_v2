"""MildTrendLongI1hV14 — Donchian Breakout + Volume Surge.

Economic logic: Donchian channel breakouts identify price reaching new highs,
signaling potential trend continuation. Volume surge confirmation (>1.3x average)
ensures the breakout is backed by real participation rather than thin-market noise.
A weaker signal fires when price holds above the channel midpoint with positive
price momentum, capturing trend continuation without requiring new highs.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV14(TrendingStrategy):
    """Donchian upper breakout with volume surge, midpoint continuation fallback.

    Signal logic:
        close > donchian_upper AND volume > vol_sma * 1.3 -> 0.8
        close > donchian_middle AND close > prev_close -> 0.4
        else -> 0.0
    """

    name = "mild_trend_long_I_1h_v14"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50  # dc_period(40) + buffer(10)

    # Optimizable parameters
    dc_period: int = 40
    vol_period: int = 30
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
        """Precompute Donchian channels and volume SMA."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_middle = donchian(
            self._highs, self._lows, period=self.dc_period,
        )

        # Compute simple moving average of volume using numpy
        n = len(self._volumes)
        self._vol_sma = np.full(n, np.nan)
        for i in range(self.vol_period - 1, n):
            window = self._volumes[i - self.vol_period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._vol_sma[i] = np.mean(valid)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal based on Donchian breakout with volume confirmation."""
        if bar_index < 1:
            return 0.0

        close = self._closes[bar_index]
        prev_close = self._closes[bar_index - 1]
        dc_upper = self._dc_upper[bar_index]
        dc_middle = self._dc_middle[bar_index]
        volume = self._volumes[bar_index]
        vol_sma = self._vol_sma[bar_index]

        if (
            np.isnan(close) or np.isnan(prev_close) or np.isnan(dc_upper)
            or np.isnan(dc_middle) or np.isnan(volume) or np.isnan(vol_sma)
        ):
            return 0.0

        # Strong signal: breakout above upper channel with volume surge
        if close > dc_upper and vol_sma > 0.0 and volume > vol_sma * 1.3:
            return 0.8

        # Weak signal: above midpoint with positive momentum
        if close > dc_middle and close > prev_close:
            return 0.4

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for auto-generated panels."""
        return [
            {"name": "Donchian Upper", "array": self._dc_upper, "style": "dash"},
            {"name": "Donchian Middle", "array": self._dc_middle},
            {"name": "Donchian Lower", "array": self._dc_lower, "style": "dash"},
            {"name": "Volume SMA", "array": self._vol_sma},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"DC Mid({self.dc_period})", datetimes, self._dc_middle, color="#ffab40"),
            self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, style="dash", color="#26a69a")
        ]
        return {"overlays": overlays, "subplots": []}
