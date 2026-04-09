"""StrongTrendLongAGDailyV4 — EMA(120,250) + MACD(70,150,50) + CMF(120).

Economic logic: Dual EMA crossover captures Silver's long-term USD/macro
regime shifts. Wide MACD parameters filter daily noise while tracking
intermediate momentum. CMF with extended lookback confirms sustained
institutional buying pressure in the Silver market.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.momentum.macd import macd
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAGDailyV4(TrendingStrategy):
    """EMA cross + MACD momentum + CMF volume confirmation.

    Signal logic:
        - Fast EMA > Slow EMA AND MACD line > 0 AND CMF > 0 → long signal
        - Strength scales with MACD magnitude and CMF

    Attributes:
        ema_fast:        Fast EMA period.
        ema_slow:        Slow EMA period.
        macd_fast:       MACD fast EMA period.
        macd_slow:       MACD slow EMA period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_AG_daily_v4"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 300

    ema_fast: int = 120
    ema_slow: int = 250
    macd_fast: int = 70
    macd_slow: int = 150
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
        self._ema_fast = ema(self._closes, period=self.ema_fast)
        self._ema_slow = ema(self._closes, period=self.ema_slow)
        self._macd_line, self._macd_signal, _ = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=50,
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes, period=120,
        )

    def _generate_signal(self, bar_index: int) -> float:
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        ml = self._macd_line[bar_index]
        c = self._cmf[bar_index]

        if any(np.isnan(v) for v in (ef, es, ml, c)):
            return 0.0

        ema_bull = ef > es
        macd_bull = ml > 0.0
        cmf_bull = c > 0.0

        if ema_bull and macd_bull and cmf_bull:
            strength = min(1.0, c * 2.0 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast, "type": "overlay"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow, "type": "overlay"},
            {"name": f"MACD({self.macd_fast},{self.macd_slow})",
             "array": self._macd_line, "type": "subplot", "zero_line": True},
            {"name": "CMF(120)", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_fast})", datetimes, self._ema_fast, color="#ffab40"),
                self._make_overlay(f"EMA({self.ema_slow})", datetimes, self._ema_slow, color="#ab47bc"),
            ],
            "subplots": [
                self._make_subplot(
                    f"MACD({self.macd_fast},{self.macd_slow})",
                    [self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#42a5f5"),
                     self._make_subplot_trace("Signal", datetimes, self._macd_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "CMF(120)",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#66bb6a")],
                    zero_line=True,
                ),
            ],
        }
