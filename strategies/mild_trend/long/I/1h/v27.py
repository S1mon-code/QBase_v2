"""MildTrendLongI1hV27 — Schaff Trend Cycle + Squeeze Probability.

Economic logic: The Schaff Trend Cycle (STC) combines MACD with double
stochastic smoothing for cycle-aware momentum detection. Readings above 50
and rising indicate bullish cycle phase. The Squeeze Probability detector
identifies conditions where high open interest, volume surges, and declining
OI coincide — signaling forced short covering (short squeeze). When STC
confirms bullish momentum AND squeeze probability is elevated, the trend is
likely to accelerate as trapped shorts are forced to cover, creating a
self-reinforcing upward price spiral in iron ore futures.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.structure.squeeze_detector import squeeze_probability
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV27(TrendingStrategy):
    """STC bullish cycle + short squeeze probability confirmation.

    Signal logic:
        STC > 50 AND rising AND short_squeeze_prob > 0.3 -> strong (0.7-1.0)
        STC > 50 AND rising                               -> moderate (0.4-0.6)
        STC > 25 AND rising                               -> weak (0.2-0.3)
        else                                              -> 0.0
    """

    name = "mild_trend_long_I_1h_v27"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80  # schaff slow(50) + 2*period + squeeze 2*period + buffer

    # Optimizable parameters
    stc_period: int = 10
    stc_slow: int = 50
    squeeze_period: int = 20
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
        self._stc = schaff_trend_cycle(
            self._closes, period=self.stc_period, slow=self.stc_slow,
        )
        self._ss_prob, self._ls_prob = squeeze_probability(
            self._closes, self._oi, self._volumes, period=self.squeeze_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        stc_val = self._stc[bar_index]
        ss_prob = self._ss_prob[bar_index]

        if np.isnan(stc_val):
            return 0.0

        # Check STC direction
        stc_prev = self._stc[bar_index - 1] if bar_index > 0 else np.nan
        stc_rising = not np.isnan(stc_prev) and stc_val > stc_prev

        if stc_val <= 25.0 or not stc_rising:
            return 0.0

        # STC strength: scale based on level
        if stc_val > 50.0:
            stc_strength = min(1.0, (stc_val - 50.0) / 50.0)
            base_signal = 0.4 + stc_strength * 0.2
        else:
            # Between 25 and 50, early signal
            base_signal = 0.2 + (stc_val - 25.0) / 25.0 * 0.2

        # Short squeeze probability confirmation
        if not np.isnan(ss_prob) and ss_prob > 0.3:
            squeeze_boost = min(0.3, ss_prob * 0.4)
            return min(1.0, base_signal + squeeze_boost)

        return min(0.6, base_signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "Schaff Trend Cycle", "array": self._stc},
            {"name": "Short Squeeze", "array": self._ss_prob, "panel": "Squeeze Probability"},
            {"name": "Long Squeeze", "array": self._ls_prob, "panel": "Squeeze Probability"},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"STC({self.stc_period})",
                [self._make_subplot_trace("STC", datetimes, self._stc, color="#bb86fc")],
                y_range=[0, 100],
            ),
            self._make_subplot(
                f"Squeeze({self.squeeze_period})",
                [
                    self._make_subplot_trace("Short Squeeze", datetimes, self._ss_prob, color="#26a69a"),
                    self._make_subplot_trace("Long Squeeze", datetimes, self._ls_prob, color="#ef5350"),
                ],
                y_range=[0, 1],
            )
        ]
        return {"overlays": [], "subplots": subplots}
