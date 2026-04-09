"""MildTrendLongI1hV24 — KAMA (Kaufman Adaptive MA) + MFI.

Economic logic: KAMA adapts its smoothing speed based on the Efficiency
Ratio — fast in trending markets, slow in choppy ones. When price is above
KAMA and KAMA slope is positive, the market is in a confirmed adaptive
uptrend. MFI (Money Flow Index) is a volume-weighted RSI that incorporates
both price and volume, reading above 50 confirming bullish money flow.
Together they filter out false breakouts in iron ore's volatile 1H timeframe
by requiring both adaptive trend confirmation and real money backing the move.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV24(TrendingStrategy):
    """KAMA uptrend + MFI bullish money flow confirmation.

    Signal logic:
        close > KAMA AND KAMA rising AND MFI > 50 -> strong (0.7-1.0)
        close > KAMA AND KAMA rising               -> weak (0.3-0.5)
        else                                       -> 0.0
    """

    name = "mild_trend_long_I_1h_v24"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 45  # kama_period(30) + buffer(15)

    # Optimizable parameters
    kama_period: int = 20
    mfi_period: int = 14
    kama_fast_sc: int = 2
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
        self._kama = kama(
            self._closes, period=self.kama_period, fast_sc=self.kama_fast_sc,
        )
        self._mfi = mfi(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.mfi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        kama_val = self._kama[bar_index]
        close_val = self._closes[bar_index]
        mfi_val = self._mfi[bar_index]

        if np.isnan(kama_val) or np.isnan(close_val):
            return 0.0

        # Price must be above KAMA
        if close_val <= kama_val:
            return 0.0

        # KAMA must be rising
        kama_prev = self._kama[bar_index - 1] if bar_index > 0 else np.nan
        if np.isnan(kama_prev) or kama_val <= kama_prev:
            return 0.0

        # Distance above KAMA as trend strength
        distance_pct = (close_val - kama_val) / kama_val
        base_signal = min(0.5, 0.3 + distance_pct * 10.0)

        # MFI confirmation
        if not np.isnan(mfi_val) and mfi_val > 50.0:
            mfi_strength = min(0.4, (mfi_val - 50.0) / 50.0 * 0.4)
            return min(1.0, base_signal + 0.2 + mfi_strength)

        return min(0.5, base_signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"MFI({self.mfi_period})",
                [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#bb86fc")],
                horizontal_lines=[20, 80], y_range=[0, 100],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}

