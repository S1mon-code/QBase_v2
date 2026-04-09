"""MildTrendLongIDailyV27 — Vortex Indicator + Klinger Volume Oscillator.

Economic logic: The Vortex Indicator separates positive (VI+) and negative (VI-)
trend movement using true range geometry. When VI+ > VI- the market is in a
bullish vortex; the reverse indicates a bearish vortex. The Klinger Volume
Oscillator (KVO) measures the long-term trend of money flow by combining price
direction and volume. KVO crossing its signal line in the same direction as the
Vortex reading increases confidence; disagreement reduces the signal strength.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.vortex import vortex
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV27(TrendingStrategy):
    """Vortex indicator confirmed by Klinger Volume Oscillator.

    Signal logic:
        - VI+ > VI- AND KVO > KVO_signal: +1.0 (full long)
        - VI- > VI+ AND KVO < KVO_signal: -1.0 (full short)
        - Indicators disagree: ±0.4 (weak signal in Vortex direction)

    Attributes:
        vortex_period:   Vortex Indicator lookback period.
        klinger_fast:    Klinger fast EMA period.
        klinger_slow:    Klinger slow EMA period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v27"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 89  # klinger_slow(55) + vortex_period(14) + 20

    vortex_period: int = 14
    klinger_fast: int = 34
    klinger_slow: int = 55
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
        """Precompute Vortex and Klinger arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._vi_plus, self._vi_minus = vortex(
            self._highs,
            self._lows,
            self._closes,
            period=self.vortex_period,
        )
        self._kvo, self._kvo_signal = klinger(
            self._highs,
            self._lows,
            self._closes,
            self._volumes,
            fast=self.klinger_fast,
            slow=self.klinger_slow,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on Vortex and Klinger agreement."""
        vi_p = self._vi_plus[bar_index]
        vi_m = self._vi_minus[bar_index]
        kvo = self._kvo[bar_index]
        kvo_sig = self._kvo_signal[bar_index]

        if np.isnan(vi_p) or np.isnan(vi_m) or np.isnan(kvo) or np.isnan(kvo_sig):
            return 0.0

        vortex_sign = 1.0 if vi_p > vi_m else -1.0
        kvo_ok = (vortex_sign > 0 and kvo > kvo_sig) or (
            vortex_sign < 0 and kvo < kvo_sig
        )
        return vortex_sign if kvo_ok else vortex_sign * 0.4

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "vortex", "params": {"period": self.vortex_period}},
            {
                "name": "klinger",
                "params": {
                    "fast": self.klinger_fast,
                    "slow": self.klinger_slow,
                },
            },
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"Vortex({self.vortex_period})",
                [
                    self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#26a69a"),
                    self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                ],
            ),
            self._make_subplot(
                f"Klinger({self.klinger_fast})",
                [
                    self._make_subplot_trace("KVO", datetimes, self._kvo, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._kvo_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
