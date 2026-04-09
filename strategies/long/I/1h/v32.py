"""MildTrendLongI1hV32 — Stochastic K/D + OBV EMA Direction.

Economic logic: Stochastic crossovers catch momentum shifts early, but false
signals are common in low-volume drift.  Requiring OBV to trade above its own
EMA filters out moves lacking institutional participation, exploiting retail
traders who chase price-only signals without volume confirmation.
"""
from __future__ import annotations

import numpy as np

from indicators._utils import _ema
from indicators.momentum.stochastic import stochastic
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV32(TrendingStrategy):
    """Stochastic K/D crossover confirmed by OBV trend direction.

    Signal logic:
        K > D AND OBV > OBV_EMA → long = (K - D) / 100.
        K < D AND OBV < OBV_EMA → short = -(D - K) / 100.
        Disagree → 0.
    """

    name = "long_I_1h_v32"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 54  # k_period + obv_ema_period + 20

    # Optimizable parameters
    k_period: int = 14
    d_period: int = 3
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
        """Precompute Stochastic K/D and OBV with its EMA."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._k, self._d = stochastic(
            self._highs, self._lows, self._closes,
            k_period=self.k_period, d_period=self.d_period,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_ema = _ema(self._obv, self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from Stochastic + OBV agreement."""
        k_val = self._k[bar_index]
        d_val = self._d[bar_index]
        obv_val = self._obv[bar_index]
        obv_ema_val = self._obv_ema[bar_index]

        if (
            np.isnan(k_val) or np.isnan(d_val)
            or np.isnan(obv_val) or np.isnan(obv_ema_val)
        ):
            return 0.0

        stoch_bull = k_val > d_val
        obv_bull = obv_val > obv_ema_val

        if stoch_bull and obv_bull:
            return min(1.0, (k_val - d_val) / 100.0)
        if not stoch_bull and not obv_bull:
            return -min(1.0, (d_val - k_val) / 100.0)

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "stochastic", "params": {"k_period": self.k_period, "d_period": self.d_period}},
            {"name": "obv_ema", "params": {"obv_ema_period": self.obv_ema_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"OBV EMA({self.obv_ema_period})",
                [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#bb86fc")],
            )
        ]
        return {"overlays": [], "subplots": subplots}

