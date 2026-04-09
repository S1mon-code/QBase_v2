"""MildTrendLongI1hV26 — Hull Moving Average + Volume Spike.

Economic logic: The Hull Moving Average (HMA) dramatically reduces lag
compared to standard MAs by using weighted moving averages of different
periods combined via sqrt-period smoothing. When price is above a rising
HMA, the trend is confirmed with minimal delay. Volume spikes (bars where
volume exceeds a threshold multiple of the rolling average) indicate
institutional participation or breakout events. A rising HMA plus a
volume spike signals a high-conviction trend acceleration in iron ore.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV26(TrendingStrategy):
    """HMA uptrend + volume spike event confirmation.

    Signal logic:
        close > HMA AND HMA rising AND volume spike -> strong (0.8-1.0)
        close > HMA AND HMA rising                  -> moderate (0.4-0.6)
        close > HMA only                            -> weak (0.2-0.3)
        else                                        -> 0.0
    """

    name = "mild_trend_long_I_1h_v26"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50  # hma_period(30) + sqrt smoothing + vol spike lookback + buffer

    # Optimizable parameters
    hma_period: int = 30
    spike_period: int = 20
    spike_threshold: float = 2.0
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
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._vol_spike = volume_spike(
            self._volumes, period=self.spike_period, threshold=self.spike_threshold,
        )

    def _generate_signal(self, bar_index: int) -> float:
        hma_val = self._hma[bar_index]
        close_val = self._closes[bar_index]
        is_spike = self._vol_spike[bar_index]

        if np.isnan(hma_val) or np.isnan(close_val):
            return 0.0

        # Price must be above HMA
        if close_val <= hma_val:
            return 0.0

        # Check if HMA is rising
        hma_prev = self._hma[bar_index - 1] if bar_index > 0 else np.nan
        hma_rising = not np.isnan(hma_prev) and hma_val > hma_prev

        # Distance above HMA as strength
        distance_pct = (close_val - hma_val) / hma_val

        if hma_rising:
            base_signal = min(0.6, 0.4 + distance_pct * 5.0)
            if is_spike:
                # Volume spike during rising HMA = strong breakout
                return min(1.0, base_signal + 0.3)
            return base_signal

        # Price above HMA but not rising — weak
        return min(0.3, 0.2 + distance_pct * 3.0)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma},
            {"name": "Volume Spike", "array": self._vol_spike.astype(float), "style": "bar"},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"Vol Spike({self.spike_period})",
                [self._make_subplot_trace("Vol Spike", datetimes, self._vol_spike, style="bar", color_positive="#26a69a", color_negative="#ef5350")],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
