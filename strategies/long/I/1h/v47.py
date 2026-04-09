"""MildTrendLongI1hV47 — Ergodic Oscillator + CMF.

Economic logic: The Ergodic Oscillator (True Strength Index variant) measures
momentum through double-smoothed price changes, producing cleaner signals than
raw momentum. Chaikin Money Flow confirms that volume-weighted accumulation or
distribution aligns with the oscillator's direction, filtering noise in
iron-ore's often volatile 1h bars.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.ergodic import ergodic
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV47(TrendingStrategy):
    """Ergodic oscillator direction confirmed by CMF.

    Signal logic:
        - Ergodic > signal AND CMF > 0: +min(1.0, abs(ergodic) / 30)
        - Ergodic < signal AND CMF < 0: -min(1.0, abs(ergodic) / 30)
        - Disagreement: 0.0
    """

    name = "long_I_1h_v47"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60  # ergo_long(20) + cmf_period(20) + 20

    ergo_short: int = 5
    ergo_long: int = 20
    cmf_period: int = 20
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
        """Precompute Ergodic and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ergo_line, self._ergo_signal = ergodic(
            self._closes,
            short_period=self.ergo_short,
            long_period=self.ergo_long,
            signal_period=5,
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on Ergodic and CMF agreement."""
        ergo_val = self._ergo_line[bar_index]
        ergo_sig = self._ergo_signal[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(ergo_val) or np.isnan(ergo_sig) or np.isnan(cmf_val):
            return 0.0

        strength = min(1.0, abs(ergo_val) / 30.0)

        if ergo_val > ergo_sig and cmf_val > 0:
            return strength
        if ergo_val < ergo_sig and cmf_val < 0:
            return -strength
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "ergodic",
                "params": {
                    "short_period": self.ergo_short,
                    "long_period": self.ergo_long,
                    "signal_period": 5,
                },
            },
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                f"Ergodic({self.ergo_short},{self.ergo_long})",
                [
                    self._make_subplot_trace("Ergodic", datetimes, self._ergo_line, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._ergo_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}

