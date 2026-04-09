"""MildTrendLongI2hV20 — EMA Ribbon (10/30/60) + CMF Confirmation.

Economic logic: An EMA ribbon with three periods captures trend alignment across
multiple timeframes. Perfect alignment (EMA10 > EMA30 > EMA60) indicates that
short, medium, and long-term momentum all agree — a high-confidence trend state.
CMF confirms that volume-weighted money flow supports the aligned direction,
filtering out trends driven by thin volume.
"""

from __future__ import annotations

import numpy as np

from indicators._utils import _ema
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV20(TrendingStrategy):
    """EMA ribbon alignment with CMF volume confirmation.

    Signal logic:
        - EMA10 > EMA30 > EMA60 AND CMF > 0 → +1.0
        - EMA10 < EMA30 < EMA60 AND CMF < 0 → -1.0
        - Aligned but no CMF confirm → 0.7 * direction
        - Not aligned → 0.0

    Attributes:
        ema_fast:        Fast EMA period.
        ema_mid:         Medium EMA period.
        ema_slow:        Slow EMA period.
        cmf_period:      Chaikin Money Flow lookback period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_2h_v20"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 100  # ema_slow + cmf_period + 20

    ema_fast: int = 10
    ema_mid: int = 30
    ema_slow: int = 60
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
        """Precompute EMA ribbon and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = _ema(self._closes, self.ema_fast)
        self._ema_mid = _ema(self._closes, self.ema_mid)
        self._ema_slow = _ema(self._closes, self.ema_slow)
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return EMA ribbon alignment signal confirmed by CMF."""
        ef = self._ema_fast[bar_index]
        em = self._ema_mid[bar_index]
        es = self._ema_slow[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(ef) or np.isnan(em) or np.isnan(es) or np.isnan(cmf_val):
            return 0.0

        bullish_aligned = ef > em > es
        bearish_aligned = ef < em < es

        if bullish_aligned:
            if cmf_val > 0.0:
                return 1.0
            return 0.7
        if bearish_aligned:
            if cmf_val < 0.0:
                return -1.0
            return -0.7
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "ema_ribbon",
                "params": {
                    "fast": self.ema_fast,
                    "mid": self.ema_mid,
                    "slow": self.ema_slow,
                },
            },
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]
