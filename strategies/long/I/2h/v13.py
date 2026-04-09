"""MildTrendLongI2hV13 — Momentum Acceleration + CMF Confirmation.

Economic logic: Momentum acceleration measures the rate of change of momentum
itself — positive acceleration means trends are strengthening, negative means
weakening. Chaikin Money Flow confirms that institutional money flow aligns
with the acceleration direction, filtering out momentum bursts without volume
backing.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.momentum_accel import momentum_acceleration
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV13(TrendingStrategy):
    """Momentum acceleration with CMF volume confirmation.

    Signal logic:
        - Accelerating (mom_accel > 0) AND CMF > 0 → +min(1.0, mom_accel * 10)
        - Decelerating (mom_accel < 0) AND CMF < 0 → -min(1.0, abs(mom_accel) * 10)
        - Disagree → 0.0

    Attributes:
        accel_fast:      Fast momentum period.
        accel_slow:      Slow momentum period.
        cmf_period:      Chaikin Money Flow lookback period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_2h_v13"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60  # accel_slow + cmf_period + 20

    accel_fast: int = 10
    accel_slow: int = 20
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
        """Precompute momentum acceleration and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._mom_accel = momentum_acceleration(
            self._closes, fast_period=self.accel_fast, slow_period=self.accel_slow,
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return momentum acceleration signal confirmed by CMF."""
        accel_val = self._mom_accel[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(accel_val) or np.isnan(cmf_val):
            return 0.0

        if accel_val > 0.0 and cmf_val > 0.0:
            return min(1.0, accel_val * 10.0)
        if accel_val < 0.0 and cmf_val < 0.0:
            return -min(1.0, abs(accel_val) * 10.0)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "momentum_acceleration",
                "params": {
                    "fast_period": self.accel_fast,
                    "slow_period": self.accel_slow,
                },
            },
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]
