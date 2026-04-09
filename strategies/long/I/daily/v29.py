"""MildTrendLongIDailyV29 — Aroon Oscillator Strength + CMF Filter.

Economic logic: The Aroon Oscillator measures trend strength and recency of
highs/lows. Values above +60 indicate a strong, sustained uptrend; below -60
indicate a strong downtrend. The ±30 dead zone captures sideways/unclear
conditions and returns no signal. Chaikin Money Flow (CMF) acts as a volume
filter: CMF > 0 confirms buying pressure, CMF < 0 confirms selling pressure.
When CMF disagrees with the Aroon direction, signal strength is halved.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV29(TrendingStrategy):
    """Aroon oscillator strength filtered by Chaikin Money Flow.

    Signal logic:
        - |aroon_osc| < 30: 0.0 (sideways — no signal)
        - aroon_osc > 60 AND CMF > 0:  +1.0
        - aroon_osc 30-60 AND CMF > 0: +0.6
        - aroon_osc -60 to -30 AND CMF < 0: -0.6
        - aroon_osc < -60 AND CMF < 0:  -1.0
        - CMF disagrees with Aroon direction: 50% of base signal

    Attributes:
        aroon_period:    Aroon lookback period.
        cmf_period:      Chaikin Money Flow lookback period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v29"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 65  # aroon_period(25) + cmf_period(20) + 20

    aroon_period: int = 25
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
        """Precompute Aroon oscillator and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        _, _, self._aroon_osc = aroon(
            self._highs,
            self._lows,
            period=self.aroon_period,
        )
        self._cmf = cmf(
            self._highs,
            self._lows,
            self._closes,
            self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on Aroon strength and CMF filter."""
        aroon_osc = self._aroon_osc[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(aroon_osc) or np.isnan(cmf_val):
            return 0.0

        if abs(aroon_osc) < 30:
            return 0.0

        strength = 1.0 if abs(aroon_osc) > 60 else 0.6
        base = 1.0 if aroon_osc > 0 else -1.0
        cmf_ok = (base > 0 and cmf_val > 0) or (base < 0 and cmf_val < 0)
        return base * strength if cmf_ok else base * strength * 0.5

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "aroon", "params": {"period": self.aroon_period}},
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
                f"Aroon Osc({self.aroon_period})",
                [self._make_subplot_trace("Aroon Osc", datetimes, self._aroon_osc, color="#4fc3f7")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
