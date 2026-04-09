"""MildTrendLongI4hV30 — Momentum Acceleration + OBV + ATR Filter.

Economic logic: Momentum Acceleration measures whether price momentum is
increasing or decaying. OBV tracks cumulative volume flow. ATR expansion
confirms volatility is supporting the move. In iron ore, the confluence of
accelerating momentum, positive volume flow, and expanding volatility marks
the strongest trend phases.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.momentum_accel import momentum_acceleration
from indicators.volume.obv import obv
from indicators._utils import _ema, _sma
from indicators.volatility.atr import atr
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV30(TrendingStrategy):
    """Momentum Acceleration + OBV trend + ATR expansion filter.

    Signal logic:
        - mom_accel > 0 AND OBV > OBV_EMA AND ATR > ATR_SMA -> +min(1.0, accel * 10)
        - All bearish -> -min(1.0, abs(accel) * 10)
        - 2-of-3 agree -> +/-0.4
        - ATR not expanding -> signal * 0.3
        - Otherwise -> 0.0
    """

    name = "mild_trend_long_I_4h_v30"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60  # accel_slow(20) + obv_ema_period(20) + 20

    accel_fast: int = 10
    accel_slow: int = 20
    obv_ema_period: int = 20
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
        self._accel = momentum_acceleration(
            self._closes, fast_period=self.accel_fast, slow_period=self.accel_slow,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_ema = _ema(self._obv, self.obv_ema_period)
        self._atr = atr(self._highs, self._lows, self._closes, period=14)
        self._atr_sma = _sma(self._atr, self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        acc = self._accel[bar_index]
        obv_val = self._obv[bar_index]
        obv_ema = self._obv_ema[bar_index]
        atr_val = self._atr[bar_index]
        atr_avg = self._atr_sma[bar_index]

        if np.isnan(acc) or np.isnan(obv_val) or np.isnan(obv_ema) or np.isnan(atr_val) or np.isnan(atr_avg):
            return 0.0

        mom_bull = acc > 0
        obv_bull = obv_val > obv_ema
        atr_expand = atr_val > atr_avg

        bull_count = int(mom_bull) + int(obv_bull) + int(atr_expand)
        bear_count = int(not mom_bull) + int(not obv_bull) + int(atr_expand)

        strength = min(1.0, abs(acc) * 10.0)

        # All three bullish
        if mom_bull and obv_bull and atr_expand:
            return strength
        # All three bearish (mom down, obv down, atr expanding)
        if not mom_bull and not obv_bull and atr_expand:
            return -strength
        # 2-of-3 agree
        if bull_count >= 2 and mom_bull:
            sig = 0.4
            if not atr_expand:
                sig *= 0.3
            return sig
        if bear_count >= 2 and not mom_bull:
            sig = -0.4
            if not atr_expand:
                sig *= 0.3
            return sig
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "momentum_acceleration", "params": {"fast": self.accel_fast, "slow": self.accel_slow}},
            {"name": "obv", "params": {"ema_period": self.obv_ema_period}},
            {"name": "atr", "params": {"period": 14}},
        ]
