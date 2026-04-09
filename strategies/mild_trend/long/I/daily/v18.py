"""MildTrendLongIDailyV18 — HMA Direction + Klinger Volume Oscillator.

Economic logic: The Hull Moving Average minimises lag while retaining
smoothness, making its direction changes a reliable trend signal.
The Klinger Oscillator identifies volume-driven money flow divergence
from price — agreement between HMA direction and Klinger confirms
institutional participation in the trend.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV18(TrendingStrategy):
    """HMA direction + Klinger Volume Oscillator (Medium Horizon).

    Signal logic:
        - HMA rising AND KVO > signal line  →  +1.0
        - HMA falling AND KVO < signal line  →  -1.0
        - Disagreement between HMA and Klinger  →  0.4 * HMA direction

    Attributes:
        hma_period:      Period for Hull Moving Average.
        klinger_fast:    Fast EMA period for Klinger oscillator.
        klinger_slow:    Slow EMA period for Klinger oscillator.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v18"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 90

    hma_period: int = 40
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
        """Precompute HMA and Klinger arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._kvo, self._kvo_signal = klinger(
            self._highs,
            self._lows,
            self._closes,
            self._volumes,
            fast=self.klinger_fast,
            slow=self.klinger_slow,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on HMA direction and Klinger confirmation."""
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1] if bar_index > 0 else h
        kvo = self._kvo[bar_index]
        kvo_sig = self._kvo_signal[bar_index]

        if np.isnan(h) or np.isnan(h_prev) or np.isnan(kvo) or np.isnan(kvo_sig):
            return 0.0

        hma_sign = 1.0 if h > h_prev else -1.0 if h < h_prev else 0.0

        if hma_sign == 0.0:
            return 0.0

        klinger_ok = (hma_sign > 0 and kvo > kvo_sig) or (hma_sign < 0 and kvo < kvo_sig)

        return hma_sign if klinger_ok else hma_sign * 0.4

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "hma", "params": {"period": self.hma_period}},
            {
                "name": "klinger",
                "params": {"fast": self.klinger_fast, "slow": self.klinger_slow},
            },
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"Klinger({self.klinger_fast})",
                [
                    self._make_subplot_trace("KVO", datetimes, self._kvo, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._kvo_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
