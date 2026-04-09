"""MildTrendLongI4hV21 — Triple EMA Slopes (20/40/80) + CMF Confirmation.

Economic logic: When short, medium, and long-term EMAs all slope in the same
direction, a strong intermediate trend is in place. CMF confirms that volume
is flowing in the trend direction, filtering false trend signals.
"""

from __future__ import annotations

import numpy as np

from indicators._utils import _ema
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV21(TrendingStrategy):
    """Triple EMA slope alignment confirmed by Chaikin Money Flow.

    Signal logic:
        - All 3 EMA slopes up AND CMF > 0 -> +1.0
        - All 3 EMA slopes down AND CMF < 0 -> -1.0
        - 2-of-3 slopes agree AND CMF agrees -> +/-0.5
        - Otherwise -> 0.0
    """

    name = "mild_trend_long_I_4h_v21"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 120  # ema_slow(80) + cmf_period(20) + 20

    ema_fast: int = 20
    ema_mid: int = 40
    ema_slow: int = 80
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
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        ema_f = _ema(self._closes, self.ema_fast)
        ema_m = _ema(self._closes, self.ema_mid)
        ema_s = _ema(self._closes, self.ema_slow)
        self._slope_fast = np.diff(ema_f, prepend=np.nan)
        self._slope_mid = np.diff(ema_m, prepend=np.nan)
        self._slope_slow = np.diff(ema_s, prepend=np.nan)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        sf = self._slope_fast[bar_index]
        sm = self._slope_mid[bar_index]
        ss = self._slope_slow[bar_index]
        cv = self._cmf[bar_index]

        if np.isnan(sf) or np.isnan(sm) or np.isnan(ss) or np.isnan(cv):
            return 0.0

        up_count = int(sf > 0) + int(sm > 0) + int(ss > 0)
        dn_count = int(sf < 0) + int(sm < 0) + int(ss < 0)

        if up_count == 3 and cv > 0:
            return 1.0
        if dn_count == 3 and cv < 0:
            return -1.0
        if up_count >= 2 and cv > 0:
            return 0.5
        if dn_count >= 2 and cv < 0:
            return -0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "ema", "params": {"periods": [self.ema_fast, self.ema_mid, self.ema_slow]}},
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]
