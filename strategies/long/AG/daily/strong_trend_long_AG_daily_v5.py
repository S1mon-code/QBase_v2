"""StrongTrendLongAGDailyV5 — FRAMA(140) + Schaff(120,70,35) + ForceIndex(80).

Economic logic: FRAMA adapts to Silver's fractal structure while Schaff Trend
Cycle applies double stochastic smoothing to MACD, producing cleaner cycle
signals for AG daily. Force Index combines price direction and volume to
confirm Elder-style trend strength.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.frama import frama
from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAGDailyV5(TrendingStrategy):
    """FRAMA trend + Schaff cycle + Force Index confirmation.

    Signal logic:
        - FRAMA rising AND Schaff > 50 AND Force Index > 0 → long signal

    Attributes:
        frama_period:    FRAMA lookback.
        schaff_period:   Schaff stochastic period.
        schaff_fast:     Schaff MACD fast EMA.
        schaff_slow:     Schaff MACD slow EMA.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_AG_daily_v5"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 220

    frama_period: int = 140
    schaff_period: int = 120
    schaff_fast: int = 70
    schaff_slow: int = 35
    chandelier_mult: float = 3.5

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
        self._frama = frama(self._closes, period=self.frama_period)
        self._schaff = schaff_trend_cycle(
            self._closes, period=self.schaff_period,
            fast=self.schaff_fast, slow=self.schaff_slow,
        )
        self._force = force_index(self._closes, self._volumes, period=80)

    def _generate_signal(self, bar_index: int) -> float:
        f_val = self._frama[bar_index]
        f_prev = self._frama[bar_index - 1]
        sc = self._schaff[bar_index]
        fi = self._force[bar_index]

        if any(np.isnan(v) for v in (f_val, f_prev, sc, fi)):
            return 0.0

        frama_rising = f_val > f_prev
        schaff_bull = sc > 50.0
        force_bull = fi > 0.0

        if frama_rising and schaff_bull and force_bull:
            strength = min(1.0, (sc - 50.0) / 50.0 * 0.6 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"FRAMA({self.frama_period})", "array": self._frama, "type": "overlay"},
            {"name": f"Schaff({self.schaff_period},{self.schaff_fast},{self.schaff_slow})",
             "array": self._schaff, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [25, 50, 75]},
            {"name": "ForceIndex(80)", "array": self._force, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"FRAMA({self.frama_period})", datetimes, self._frama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Schaff({self.schaff_period})",
                    [self._make_subplot_trace("STC", datetimes, self._schaff, color="#ab47bc")],
                    y_range=[0, 100], horizontal_lines=[25, 50, 75],
                ),
                self._make_subplot(
                    "ForceIndex(80)",
                    [self._make_subplot_trace("Force", datetimes, self._force, color="#66bb6a")],
                    zero_line=True,
                ),
            ],
        }
