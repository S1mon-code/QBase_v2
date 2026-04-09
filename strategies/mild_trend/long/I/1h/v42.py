"""MildTrendLongI1hV42 — DEMA Crossover + CMF Confirmation.

Economic logic: Double Exponential Moving Average crossover (fast/slow) captures
intermediate trend shifts with less lag than standard EMA. Chaikin Money Flow
confirms that accumulation/distribution pressure aligns with the crossover
direction, filtering false signals in choppy iron-ore markets.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.dema import dema
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV42(TrendingStrategy):
    """DEMA fast/slow crossover confirmed by CMF.

    Signal logic:
        - DEMA_fast > DEMA_slow AND CMF > 0: +1.0
        - DEMA_fast < DEMA_slow AND CMF < 0: -1.0
        - Trend direction set but CMF disagrees: 0.5 * sign
        - All disagree: 0.0
    """

    name = "mild_trend_long_I_1h_v42"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["trend", "volume"]
    warmup: int = 70  # dema_slow(30) + cmf_period(20) + 20

    dema_fast: int = 10
    dema_slow: int = 30
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
        """Precompute DEMA fast, DEMA slow, and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dema_fast = dema(self._closes, period=self.dema_fast)
        self._dema_slow = dema(self._closes, period=self.dema_slow)
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on DEMA crossover and CMF."""
        df = self._dema_fast[bar_index]
        ds = self._dema_slow[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(df) or np.isnan(ds) or np.isnan(cmf_val):
            return 0.0

        if df > ds:
            return 1.0 if cmf_val > 0 else 0.5
        if df < ds:
            return -1.0 if cmf_val < 0 else -0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "dema_fast", "params": {"period": self.dema_fast}},
            {"name": "dema_slow", "params": {"period": self.dema_slow}},
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"DEMA({self.dema_fast})", datetimes, self._dema_fast, color="#ffab40"),
            self._make_overlay(f"DEMA({self.dema_slow})", datetimes, self._dema_slow, color="#ab47bc")
        ]
        subplots = [
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}

