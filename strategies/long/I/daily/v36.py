"""MildTrendLongIDailyV36 — MACD Line + Klinger KVO (Dual Momentum).

Economic logic: MACD line measures intermediate-term price momentum via
EMA divergence. Klinger Volume Oscillator captures volume-driven money flow
momentum. When both agree in direction, the trend signal is high-conviction.
Disagreement produces a reduced signal, respecting the dominant price driver
(MACD line).
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV36(TrendingStrategy):
    """MACD line direction confirmed by Klinger Volume Oscillator.

    Signal logic:
        - MACD line > 0 AND KVO > KVO signal: +1.0 (strong bull)
        - MACD line > 0 AND KVO < KVO signal: +0.4 (weak bull)
        - MACD line < 0 AND KVO < KVO signal: -1.0 (strong bear)
        - MACD line < 0 AND KVO > KVO signal: -0.4 (weak bear)

    Attributes:
        macd_fast:       MACD fast EMA period.
        macd_slow:       MACD slow EMA period.
        macd_signal:     MACD signal line period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v36"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 110  # klinger_slow(55) + macd_signal(9) + macd_slow(26) + 20

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
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
        """Precompute MACD line and Klinger KVO arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._macd_line, _, _ = macd(
            self._closes,
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
        )
        self._kvo, self._kvo_signal = klinger(
            self._highs,
            self._lows,
            self._closes,
            self._volumes,
            fast=34,
            slow=55,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on MACD line and KVO agreement."""
        macd_val = self._macd_line[bar_index]
        kvo = self._kvo[bar_index]
        kvo_sig = self._kvo_signal[bar_index]

        if np.isnan(macd_val) or np.isnan(kvo) or np.isnan(kvo_sig):
            return 0.0

        macd_up = macd_val > 0
        kvo_up = kvo > kvo_sig

        if macd_up and kvo_up:
            return 1.0
        if not macd_up and not kvo_up:
            return -1.0
        return 0.4 if macd_up else -0.4

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "macd",
                "params": {
                    "fast": self.macd_fast,
                    "slow": self.macd_slow,
                    "signal": self.macd_signal,
                },
            },
            {"name": "klinger", "params": {"fast": 34, "slow": 55}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"MACD({self.macd_fast},{self.macd_slow})",
                [self._make_subplot_trace("MACD Line", datetimes, self._macd_line, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                "Klinger",
                [
                    self._make_subplot_trace("KVO", datetimes, self._kvo, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._kvo_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
