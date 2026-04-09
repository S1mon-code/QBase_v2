"""MildTrendLongI1hV40 — TSI + Klinger Oscillator.

Economic logic: True Strength Index double-smooths momentum, filtering out
noise while preserving genuine trend direction.  The Klinger Volume Oscillator
measures the force of volume flowing in and out of a security.  When both
momentum (TSI) and volume flow (Klinger) agree on direction, the trend has
institutional backing.  We profit from traders who fight trends that have both
momentum and volume confirmation.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV40(TrendingStrategy):
    """TSI line/signal + Klinger oscillator/signal agreement.

    Signal logic:
        TSI > TSI_signal AND KVO > KVO_signal → +min(1.0, |tsi|/25)
        TSI < TSI_signal AND KVO < KVO_signal → -min(1.0, |tsi|/25)
        Disagree → 0.0
    """

    name = "mild_trend_long_I_1h_v40"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 45  # tsi_long + 20

    # Optimizable parameters
    tsi_long: int = 25
    tsi_short: int = 13
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
        """Precompute TSI and Klinger arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tsi_line, self._tsi_signal = tsi(
            self._closes,
            long_period=self.tsi_long,
            short_period=self.tsi_short,
        )
        self._kvo, self._kvo_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from TSI + Klinger agreement."""
        tsi_val = self._tsi_line[bar_index]
        tsi_sig = self._tsi_signal[bar_index]
        kvo_val = self._kvo[bar_index]
        kvo_sig = self._kvo_signal[bar_index]

        if (
            np.isnan(tsi_val) or np.isnan(tsi_sig)
            or np.isnan(kvo_val) or np.isnan(kvo_sig)
        ):
            return 0.0

        tsi_bull = tsi_val > tsi_sig
        kvo_bull = kvo_val > kvo_sig

        strength = min(1.0, abs(tsi_val) / 25.0)

        if tsi_bull and kvo_bull:
            return strength
        if not tsi_bull and not kvo_bull:
            return -strength

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "tsi", "params": {"long_period": self.tsi_long, "short_period": self.tsi_short}},
            {"name": "klinger", "params": {"fast": self.klinger_fast}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"Klinger({self.klinger_fast})",
                [
                    self._make_subplot_trace("KVO", datetimes, self._kvo, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._kvo_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            ),
            self._make_subplot(
                f"TSI({self.tsi_long},{self.tsi_short})",
                [
                    self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._tsi_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}

