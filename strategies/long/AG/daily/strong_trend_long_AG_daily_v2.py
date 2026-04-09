"""StrongTrendLongAGDailyV2 — MESA/FRAMA(140) + Coppock(150,200,80) + OBV(150).

Economic logic: FRAMA adapts to Silver's fractal dimension, widening during
volatile macro moves and tightening during consolidation. Coppock Curve captures
long-term momentum bottoms ideal for daily Silver. OBV trend confirms
accumulation by large players.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.frama import frama
from indicators.momentum.coppock import coppock
from indicators.volume.obv import obv
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAGDailyV2(TrendingStrategy):
    """FRAMA trend + Coppock momentum + OBV accumulation.

    Signal logic:
        - FRAMA rising AND Coppock > 0 AND OBV EMA rising → long signal
        - Strength scales with Coppock magnitude

    Attributes:
        frama_period:    FRAMA lookback period.
        copp_wma:        Coppock WMA smoothing period.
        copp_roc_long:   Coppock long ROC period.
        copp_roc_short:  Coppock short ROC period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_AG_daily_v2"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 300

    frama_period: int = 140
    copp_wma: int = 80
    copp_roc_long: int = 200
    copp_roc_short: int = 150
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
        self._coppock = coppock(
            self._closes, wma_period=self.copp_wma,
            roc_long=self.copp_roc_long, roc_short=self.copp_roc_short,
        )
        raw_obv = obv(self._closes, self._volumes)
        self._obv_ema = ema(raw_obv, period=150)

    def _generate_signal(self, bar_index: int) -> float:
        f_val = self._frama[bar_index]
        f_prev = self._frama[bar_index - 1]
        cop = self._coppock[bar_index]
        obv_now = self._obv_ema[bar_index]
        obv_prev = self._obv_ema[bar_index - 1]

        if any(np.isnan(v) for v in (f_val, f_prev, cop, obv_now, obv_prev)):
            return 0.0

        frama_rising = f_val > f_prev
        coppock_bull = cop > 0.0
        obv_rising = obv_now > obv_prev

        if frama_rising and coppock_bull and obv_rising:
            strength = min(1.0, abs(cop) / 20.0 + 0.3)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"FRAMA({self.frama_period})", "array": self._frama, "type": "overlay"},
            {"name": f"Coppock({self.copp_wma},{self.copp_roc_long},{self.copp_roc_short})",
             "array": self._coppock, "type": "subplot", "zero_line": True},
            {"name": "OBV EMA(150)", "array": self._obv_ema, "type": "subplot"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"FRAMA({self.frama_period})", datetimes, self._frama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Coppock({self.copp_wma})",
                    [self._make_subplot_trace("Coppock", datetimes, self._coppock, color="#ab47bc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "OBV EMA(150)",
                    [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#66bb6a")],
                ),
            ],
        }
