"""MildTrendLongI1hV33 — CCI + CMF Agreement.

Economic logic: CCI measures deviation from a statistical mean — extreme
readings signal genuine trend acceleration.  Chaikin Money Flow validates
that smart money is actually participating.  Traders who fade CCI extremes
without checking volume flow get run over when institutions are driving price.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV33(TrendingStrategy):
    """CCI extreme-zone + CMF sign agreement.

    Signal logic:
        CCI > 100  AND CMF > 0 → +1.0
        CCI < -100 AND CMF < 0 → -1.0
        CCI in [0, 100]   AND CMF > 0 → +0.5
        CCI in [-100, 0]  AND CMF < 0 → -0.5
        Disagree → 0.0
    """

    name = "long_I_1h_v33"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 40  # max(cci_period, cmf_period) + 20

    # Optimizable parameters
    cci_period: int = 20
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
        """Precompute CCI and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return graded signal from CCI zone + CMF agreement."""
        cci_val = self._cci[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(cci_val) or np.isnan(cmf_val):
            return 0.0

        # Strong signals — CCI in extreme zone with CMF confirmation
        if cci_val > 100.0 and cmf_val > 0.0:
            return 1.0
        if cci_val < -100.0 and cmf_val < 0.0:
            return -1.0

        # Moderate signals — CCI trending with CMF confirmation
        if 0.0 <= cci_val <= 100.0 and cmf_val > 0.0:
            return 0.5
        if -100.0 <= cci_val < 0.0 and cmf_val < 0.0:
            return -0.5

        # Disagree
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "cci", "params": {"period": self.cci_period}},
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
                f"CCI({self.cci_period})",
                [self._make_subplot_trace("CCI", datetimes, self._cci, color="#4fc3f7")],
                horizontal_lines=[-100, 100],
            )
        ]
        return {"overlays": [], "subplots": subplots}

