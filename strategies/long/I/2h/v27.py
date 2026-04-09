"""MildTrendLongI2hV27 — Coppock Curve + CMF.

Economic logic: The Coppock Curve is a long-term momentum oscillator originally
designed to identify market bottoms via smoothed ROC. Combined with Chaikin
Money Flow, which measures accumulation/distribution pressure, the classic
Coppock zero-cross buy signal is confirmed by volume flow alignment.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.coppock import coppock
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV27(TrendingStrategy):
    """Coppock Curve direction confirmed by CMF.

    Signal logic:
        - Coppock crosses zero from below AND CMF > 0: +1.0 (classic buy)
        - Coppock > 0 AND CMF > 0: +min(1.0, abs(Coppock)/10)
        - Coppock < 0 AND CMF < 0: -min(1.0, abs(Coppock)/10)
        - Disagree: 0.0
    """

    name = "long_I_2h_v27"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 64  # cop_wma(10) + cop_roc_long(14) + cmf_period(20) + 20

    cop_wma: int = 10
    cop_roc_long: int = 14
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
        """Precompute Coppock Curve and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._coppock = coppock(
            self._closes, wma_period=self.cop_wma,
            roc_long=self.cop_roc_long, roc_short=11,
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on Coppock and CMF."""
        cop = self._coppock[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(cop) or np.isnan(cmf_val):
            return 0.0

        # Check for zero-cross from below (classic buy signal)
        if bar_index >= 1:
            cop_prev = self._coppock[bar_index - 1]
            if not np.isnan(cop_prev) and cop > 0 and cop_prev <= 0 and cmf_val > 0:
                return 1.0

        strength = float(np.clip(abs(cop) / 10.0, 0.0, 1.0))

        if cop > 0 and cmf_val > 0:
            return strength
        if cop < 0 and cmf_val < 0:
            return -strength
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "coppock", "params": {"wma_period": self.cop_wma, "roc_long": self.cop_roc_long, "roc_short": 11}},
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]
