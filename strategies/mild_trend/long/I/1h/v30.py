"""MildTrendLongI1hV30 — MESA Adaptive MA + Wyckoff Divergence.

Economic logic: The MESA Adaptive Moving Average (MAMA/FAMA) uses the
Hilbert Transform to measure instantaneous phase and adapt smoothing to
the dominant market cycle. MAMA crossing above FAMA signals a bullish
phase transition. Wyckoff Divergence detects when price makes new lows
but the Accumulation/Distribution line does not — indicating institutional
accumulation despite falling prices (bullish divergence). When MAMA/FAMA
confirms an uptrend AND Wyckoff detects accumulation, the trend is backed
by institutional positioning, making it a high-conviction long signal
in iron ore futures.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.mesa_adaptive_ma import mama
from indicators.volume.wyckoff_divergence import wyckoff_divergence
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV30(TrendingStrategy):
    """MAMA > FAMA uptrend + Wyckoff bullish divergence confirmation.

    Signal logic:
        MAMA > FAMA AND wyckoff bullish div recent -> strong (0.8-1.0)
        MAMA > FAMA AND MAMA rising                -> moderate (0.4-0.6)
        MAMA > FAMA only                           -> weak (0.3)
        else                                       -> 0.0
    """

    name = "mild_trend_long_I_1h_v30"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50  # MESA needs 32 warmup + wyckoff lookback + buffer

    # Optimizable parameters
    mesa_fast_limit: float = 0.5
    mesa_slow_limit: float = 0.05
    wyckoff_lookback: int = 20
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
        self._mama, self._fama = mama(
            self._closes,
            fast_limit=self.mesa_fast_limit,
            slow_limit=self.mesa_slow_limit,
        )
        self._bull_div, self._bear_div, self._ad_line = wyckoff_divergence(
            self._highs, self._lows, self._closes, self._volumes,
            lookback=self.wyckoff_lookback,
        )

    def _generate_signal(self, bar_index: int) -> float:
        mama_val = self._mama[bar_index]
        fama_val = self._fama[bar_index]

        if np.isnan(mama_val) or np.isnan(fama_val):
            return 0.0

        # MAMA must be above FAMA for bullish signal
        if mama_val <= fama_val:
            return 0.0

        # MAMA-FAMA spread as trend strength
        spread = (mama_val - fama_val) / fama_val
        base_signal = min(0.5, 0.3 + spread * 20.0)

        # Check MAMA direction
        mama_prev = self._mama[bar_index - 1] if bar_index > 0 else np.nan
        mama_rising = not np.isnan(mama_prev) and mama_val > mama_prev

        if mama_rising:
            base_signal = min(0.6, base_signal + 0.1)

        # Check for recent Wyckoff bullish divergence (within last 5 bars)
        recent_bull_div = False
        lookback_window = min(5, bar_index)
        for i in range(bar_index - lookback_window, bar_index + 1):
            if i >= 0 and self._bull_div[i] > 0.5:
                recent_bull_div = True
                break

        if recent_bull_div:
            # Wyckoff bullish divergence = institutional accumulation
            return min(1.0, base_signal + 0.3)

        return base_signal

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "MAMA", "array": self._mama},
            {"name": "FAMA", "array": self._fama},
            {"name": "A/D Line", "array": self._ad_line},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay("MAMA", datetimes, self._mama, color="#ffab40"),
            self._make_overlay("FAMA", datetimes, self._fama, color="#ab47bc")
        ]
        subplots = [
            self._make_subplot(
                "Wyckoff Divergence",
                [
                    self._make_subplot_trace("Bull Div", datetimes, self._bull_div, color="#26a69a"),
                    self._make_subplot_trace("Bear Div", datetimes, self._bear_div, color="#ef5350"),
                ],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
