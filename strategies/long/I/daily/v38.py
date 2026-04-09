"""MildTrendLongIDailyV38 — HMA Cross (Fast vs Slow) + CMF Confirmation.

Economic logic: The Hull Moving Average minimises lag while remaining smooth,
making crossovers between a fast and slow HMA a reliable trend-change signal.
Chaikin Money Flow acts as a volume-participation filter: a price trend with
supporting volume flow is more likely to persist than one driven by price
alone. Full signal when HMA and CMF agree; half-strength when they diverge.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV38(TrendingStrategy):
    """HMA crossover trend direction confirmed by Chaikin Money Flow.

    Signal logic:
        - Fast HMA > Slow HMA AND CMF > 0: +1.0 (aligned bull)
        - Fast HMA < Slow HMA AND CMF < 0: -1.0 (aligned bear)
        - HMA and CMF disagree:            hma_sign * 0.5

    Attributes:
        fast_hma:        Period for the fast Hull Moving Average.
        slow_hma:        Period for the slow Hull Moving Average.
        cmf_period:      Chaikin Money Flow lookback period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v38"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 100  # slow_hma(60) + cmf_period(20) + 20

    fast_hma: int = 20
    slow_hma: int = 60
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
        """Precompute fast HMA, slow HMA, and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma_fast = hma(self._closes, period=self.fast_hma)
        self._hma_slow = hma(self._closes, period=self.slow_hma)
        self._cmf = cmf(
            self._highs,
            self._lows,
            self._closes,
            self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] from HMA cross scaled by CMF agreement."""
        hf = self._hma_fast[bar_index]
        hs = self._hma_slow[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(hf) or np.isnan(hs) or np.isnan(cmf_val):
            return 0.0

        hma_sign = 1.0 if hf > hs else -1.0
        cmf_ok = (hma_sign > 0 and cmf_val > 0) or (hma_sign < 0 and cmf_val < 0)

        return hma_sign if cmf_ok else hma_sign * 0.5

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "hma_fast", "params": {"period": self.fast_hma}},
            {"name": "hma_slow", "params": {"period": self.slow_hma}},
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"HMA({self.fast_hma})", datetimes, self._hma_fast, color="#ffab40"),
            self._make_overlay(f"HMA({self.slow_hma})", datetimes, self._hma_slow, color="#ab47bc")
        ]
        subplots = [
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
