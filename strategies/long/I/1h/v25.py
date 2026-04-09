"""MildTrendLongI1hV25 — Trend Intensity Index + Chaikin Oscillator.

Economic logic: The Trend Intensity Index (TII) measures the proportion of
price deviations above vs below a moving average, producing a 0-100 reading
where values above 50 indicate uptrend dominance. The Chaikin Oscillator
measures the momentum of the Accumulation/Distribution line by computing
the difference between fast and slow EMAs of A/D. Positive Chaikin values
indicate accumulation pressure. When TII confirms a strong uptrend AND
Chaikin shows accumulation momentum, the price trend has genuine
supply/demand support rather than being just a technical artifact.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.trend_intensity import trend_intensity
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV25(TrendingStrategy):
    """TII uptrend + Chaikin accumulation confirmation.

    Signal logic:
        TII > 60 AND Chaikin > 0 and rising -> strong (0.7-1.0)
        TII > 50                             -> weak (0.3-0.5)
        else                                 -> 0.0
    """

    name = "long_I_1h_v25"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50  # TII needs 2*period values, + buffer

    # Optimizable parameters
    tii_period: int = 14
    chaikin_fast: int = 3
    chaikin_slow: int = 10
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
        self._tii = trend_intensity(self._closes, period=self.tii_period)
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.chaikin_fast, slow=self.chaikin_slow,
        )

    def _generate_signal(self, bar_index: int) -> float:
        tii_val = self._tii[bar_index]
        chk = self._chaikin[bar_index]

        if np.isnan(tii_val):
            return 0.0

        # TII must indicate uptrend (> 50)
        if tii_val <= 50.0:
            return 0.0

        # TII strength: scale from 50-100 to 0-1
        tii_strength = (tii_val - 50.0) / 50.0
        base_signal = min(0.5, 0.3 + tii_strength * 0.2)

        # Chaikin confirmation: positive and rising
        if not np.isnan(chk) and chk > 0.0:
            chk_prev = self._chaikin[bar_index - 1] if bar_index > 0 else np.nan
            chaikin_rising = not np.isnan(chk_prev) and chk > chk_prev

            if chaikin_rising:
                return min(1.0, base_signal + 0.3 + tii_strength * 0.2)
            return min(0.8, base_signal + 0.2)

        return min(0.5, base_signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "Trend Intensity", "array": self._tii},
            {"name": "Chaikin Oscillator", "array": self._chaikin},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"TII({self.tii_period})",
                [self._make_subplot_trace("TII", datetimes, self._tii, color="#bb86fc")],
                y_range=[0, 100],
            ),
            self._make_subplot(
                f"Chaikin({self.chaikin_fast},{self.chaikin_slow})",
                [self._make_subplot_trace("Chaikin Osc", datetimes, self._chaikin, color="#4fc3f7")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
