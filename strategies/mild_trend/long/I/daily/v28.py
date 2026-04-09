"""MildTrendLongIDailyV28 — HMA Slope + OI Flow Confirmation.

Economic logic: Hull Moving Average (HMA) reduces lag while maintaining
smoothness, making its slope a responsive measure of trend direction. OI Flow
tracks the directional accumulation of open interest weighted by price change —
when flow exceeds its signal line, new positions are being built in the trend
direction (smart money confirmation). Agreement of HMA slope and OI flow
produces a full signal; disagreement yields a reduced signal.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV28(TrendingStrategy):
    """HMA slope confirmed by OI Flow direction.

    Signal logic:
        - HMA rising AND OI flow > OI signal: +1.0 (full long)
        - HMA falling AND OI flow < OI signal: -1.0 (full short)
        - HMA flat: 0.0
        - Indicators disagree: ±0.3 (weak signal in HMA direction)

    Attributes:
        hma_period:      Hull Moving Average period.
        oi_period:       OI flow EMA signal period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v28"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 79  # hma_period(40) + oi_period(14) + 25

    hma_period: int = 40
    oi_period: int = 14
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
        """Precompute HMA and OI Flow arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes,
            self._oi,
            self._volumes,
            period=self.oi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on HMA slope and OI flow agreement."""
        if bar_index < 1:
            return 0.0

        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1]
        oi_fl = self._oi_flow[bar_index]
        oi_sig = self._oi_signal[bar_index]

        if np.isnan(h) or np.isnan(h_prev) or np.isnan(oi_fl) or np.isnan(oi_sig):
            return 0.0

        hma_sign = 1.0 if h > h_prev else -1.0 if h < h_prev else 0.0

        if hma_sign == 0.0:
            return 0.0

        oi_ok = (hma_sign > 0 and oi_fl > oi_sig) or (hma_sign < 0 and oi_fl < oi_sig)
        return hma_sign if oi_ok else hma_sign * 0.3

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "hma", "params": {"period": self.hma_period}},
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"OI Flow({self.oi_period})",
                [
                    self._make_subplot_trace("OI Flow", datetimes, self._oi_flow, color="#bb86fc"),
                    self._make_subplot_trace("OI Signal", datetimes, self._oi_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
