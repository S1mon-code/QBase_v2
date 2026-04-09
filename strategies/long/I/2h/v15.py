"""MildTrendLongI2hV15 — Aroon Oscillator + Klinger Volume Oscillator.

Economic logic: The Aroon oscillator measures how recently price made a new high
vs. a new low, providing a clean trend direction signal. The Klinger Volume
Oscillator accumulates volume based on high-low-close trend direction, acting
as a sophisticated volume confirmation. Together, they identify trends where
both price structure and volume flow agree on direction.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV15(TrendingStrategy):
    """Aroon oscillator with Klinger volume confirmation.

    Signal logic:
        - Aroon osc > 50 AND KVO > KVO signal → +(aroon_osc / 100)
        - Aroon osc < -50 AND KVO < KVO signal → -(abs(aroon_osc) / 100)
        - Aroon osc between -50..50 → signal * 0.3
        - KVO disagrees with Aroon direction → 0.0

    Attributes:
        aroon_period:    Aroon lookback period.
        klinger_fast:    Klinger fast EMA period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_2h_v15"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 79  # aroon_period + klinger_fast + 20

    aroon_period: int = 25
    klinger_fast: int = 34
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
        """Precompute Aroon and Klinger arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period,
        )
        self._kvo, self._kvo_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return Aroon directional signal confirmed by Klinger volume."""
        osc_val = self._aroon_osc[bar_index]
        kvo_val = self._kvo[bar_index]
        kvo_sig = self._kvo_signal[bar_index]

        if np.isnan(osc_val) or np.isnan(kvo_val) or np.isnan(kvo_sig):
            return 0.0

        kvo_bullish = kvo_val > kvo_sig
        kvo_bearish = kvo_val < kvo_sig

        if osc_val > 50.0:
            if not kvo_bullish:
                return 0.0
            return osc_val / 100.0
        if osc_val < -50.0:
            if not kvo_bearish:
                return 0.0
            return -(abs(osc_val) / 100.0)

        # Weak zone: -50 to 50
        if osc_val > 0.0 and kvo_bullish:
            return (osc_val / 100.0) * 0.3
        if osc_val < 0.0 and kvo_bearish:
            return (osc_val / 100.0) * 0.3
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "aroon", "params": {"period": self.aroon_period}},
            {"name": "klinger", "params": {"fast": self.klinger_fast}},
        ]
