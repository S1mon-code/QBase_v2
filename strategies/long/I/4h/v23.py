"""MildTrendLongI4hV23 — Bollinger Band Position + Aroon Oscillator.

Economic logic: Bollinger Band position measures where price sits relative to
its volatility envelope. Aroon Oscillator quantifies trend strength based on
time since recent highs/lows. Together they identify trending price at
volatility extremes.
"""

from __future__ import annotations

import numpy as np

from indicators.volatility.bollinger import bollinger_bands
from indicators.trend.aroon import aroon
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV23(TrendingStrategy):
    """BB position combined with Aroon Oscillator for trend confirmation.

    Signal logic:
        - bb_pos > 0.8 AND aroon_osc > 50 -> +(aroon_osc / 100)
        - bb_pos < 0.2 AND aroon_osc < -50 -> -(abs(aroon_osc) / 100)
        - bb_pos > 0.5 AND aroon_osc > 0 -> +0.4
        - bb_pos < 0.5 AND aroon_osc < 0 -> -0.4
        - Otherwise -> 0.0
    """

    name = "long_I_4h_v23"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 45  # max(bb_period, aroon_period) + 20

    bb_period: int = 20
    aroon_period: int = 25
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
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            self._closes, period=self.bb_period,
        )
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period,
        )
        bb_range = self._bb_upper - self._bb_lower
        with np.errstate(divide="ignore", invalid="ignore"):
            self._bb_pos = np.where(
                bb_range > 0,
                (self._closes - self._bb_lower) / bb_range,
                np.nan,
            )

    def _generate_signal(self, bar_index: int) -> float:
        bp = self._bb_pos[bar_index]
        ao = self._aroon_osc[bar_index]

        if np.isnan(bp) or np.isnan(ao):
            return 0.0

        if bp > 0.8 and ao > 50:
            return ao / 100.0
        if bp < 0.2 and ao < -50:
            return -(abs(ao) / 100.0)
        if bp > 0.5 and ao > 0:
            return 0.4
        if bp < 0.5 and ao < 0:
            return -0.4
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "bollinger_bands", "params": {"period": self.bb_period}},
            {"name": "aroon", "params": {"period": self.aroon_period}},
        ]
