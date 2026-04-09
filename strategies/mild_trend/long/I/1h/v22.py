"""MildTrendLongI1hV22 — Fisher Transform + Klinger Oscillator.

Economic logic: The Fisher Transform normalizes price into a Gaussian
distribution via inverse hyperbolic tangent, producing sharp turning-point
signals. When the Fisher line crosses above its trigger, bullish momentum
is building. The Klinger Volume Oscillator measures volume flow direction
by comparing fast and slow EMAs of volume force. When Klinger KVO is above
its signal line, smart money is flowing into the asset. Both confirming
together indicates a high-probability trend continuation with volume backing.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV22(TrendingStrategy):
    """Fisher bullish cross + Klinger volume flow confirmation.

    Signal logic:
        Fisher > trigger AND KVO > signal -> strong (0.7-1.0)
        Fisher > trigger only             -> weak (0.3-0.5)
        else                              -> 0.0
    """

    name = "mild_trend_long_I_1h_v22"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80  # klinger slow(55) + signal(13) + buffer

    # Optimizable parameters
    fisher_period: int = 10
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
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._fisher, self._fisher_trigger = fisher_transform(
            self._highs, self._lows, period=self.fisher_period
        )
        self._kvo, self._kvo_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=self.klinger_slow,
        )

    def _generate_signal(self, bar_index: int) -> float:
        fish = self._fisher[bar_index]
        trig = self._fisher_trigger[bar_index]
        kvo = self._kvo[bar_index]
        kvo_sig = self._kvo_signal[bar_index]

        if np.isnan(fish) or np.isnan(trig):
            return 0.0

        # Fisher bullish: line above trigger
        if fish <= trig:
            return 0.0

        # Fisher strength: how far above trigger
        fisher_strength = min(1.0, (fish - trig) * 0.5)
        base_signal = 0.3 + fisher_strength * 0.2

        # Klinger confirmation
        if not np.isnan(kvo) and not np.isnan(kvo_sig) and kvo > kvo_sig:
            kvo_strength = min(0.3, abs(kvo - kvo_sig) / (abs(kvo_sig) + 1e-10) * 0.3)
            return min(1.0, base_signal + 0.3 + kvo_strength)

        return min(0.5, base_signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "Fisher", "array": self._fisher, "panel": "Fisher Transform"},
            {"name": "Trigger", "array": self._fisher_trigger, "panel": "Fisher Transform"},
            {"name": "KVO", "array": self._kvo, "panel": "Klinger"},
            {"name": "Signal", "array": self._kvo_signal, "panel": "Klinger"},
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
                f"Fisher({self.fisher_period})",
                [
                    self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#bb86fc"),
                    self._make_subplot_trace("Trigger", datetimes, self._fisher_trigger, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}

