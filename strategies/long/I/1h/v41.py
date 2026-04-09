"""MildTrendLongI1hV41 — Awesome Oscillator + OI Flow.

Economic logic: The Awesome Oscillator measures market momentum via the
midpoint-price difference between fast and slow simple moving averages.
When AO aligns with Open Interest flow direction (flow vs its signal line),
it confirms that fresh positioning supports the momentum — a hallmark of
sustainable iron-ore trends on the 1h timeframe.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.awesome_oscillator import ao
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV41(TrendingStrategy):
    """Awesome Oscillator direction confirmed by OI Flow.

    Signal logic:
        - AO > 0 AND flow > signal: +min(1.0, AO / 50)
        - AO < 0 AND flow < signal: -min(1.0, abs(AO) / 50)
        - Disagreement: 0.0
    """

    name = "long_I_1h_v41"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 68  # ao_slow(34) + oi_period(14) + 20

    ao_fast: int = 5
    ao_slow: int = 34
    oi_period: int = 14
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
        """Precompute AO and OI Flow arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ao = ao(self._highs, self._lows, fast=self.ao_fast, slow=self.ao_slow)
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on AO and OI Flow agreement."""
        ao_val = self._ao[bar_index]
        flow_val = self._oi_flow[bar_index]
        sig_val = self._oi_signal[bar_index]

        if np.isnan(ao_val) or np.isnan(flow_val) or np.isnan(sig_val):
            return 0.0

        if ao_val > 0 and flow_val > sig_val:
            return min(1.0, ao_val / 50.0)
        if ao_val < 0 and flow_val < sig_val:
            return -min(1.0, abs(ao_val) / 50.0)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "ao",
                "params": {"fast": self.ao_fast, "slow": self.ao_slow},
            },
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"OI Flow({self.oi_period})",
                [
                    self._make_subplot_trace("OI Flow", datetimes, self._oi_flow, color="#bb86fc"),
                    self._make_subplot_trace("OI Signal", datetimes, self._oi_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            ),
            self._make_subplot(
                f"AO({self.ao_fast},{self.ao_slow})",
                [self._make_subplot_trace("AO", datetimes, self._ao, style="bar", color_positive="#26a69a", color_negative="#ef5350")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}

