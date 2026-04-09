"""MildTrendShortIDailyV4 — PSAR Bearish + ROC Negative + MFI Weak.

Economic logic: Parabolic SAR above price confirms bearish trend state.
Negative ROC over 25 bars validates declining price momentum. MFI below 45
indicates money flow is biased towards selling rather than buying.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.psar import psar
from indicators.momentum.roc import rate_of_change
from indicators.volume.mfi import mfi
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV4(TrendingStrategy):
    """PSAR bearish + ROC(25) < 0 + MFI(25) < 45."""

    name = "short_I_daily_v4"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    # Optimizable parameters (<=5 including chandelier_mult)
    roc_period: int = 25
    mfi_period: int = 25
    mfi_threshold: float = 45.0
    chandelier_mult: float = 4.0

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
        self._psar_vals, self._psar_dir = psar(self._highs, self._lows)
        self._roc = rate_of_change(self._closes, self.roc_period)
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes, self.mfi_period)

        # Pre-smooth raw signal
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        psar_d = self._psar_dir[bar_index]
        roc_val = self._roc[bar_index]
        mfi_val = self._mfi[bar_index]

        if any(np.isnan(v) for v in (psar_d, roc_val, mfi_val)):
            return 0.0

        # PSAR must be bearish (direction = -1)
        if psar_d > 0:
            return 0.0

        # ROC must be negative
        if roc_val >= 0:
            return 0.0

        # ROC magnitude drives base signal
        roc_strength = min(abs(roc_val) / 10.0, 1.0)
        base = -0.3 - roc_strength * 0.3  # -0.3 to -0.6

        # MFI confirmation
        if mfi_val < self.mfi_threshold:
            mfi_boost = min((self.mfi_threshold - mfi_val) / 45.0, 0.1)
            base -= mfi_boost

        return np.clip(base, -0.7, 0.0)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "PSAR", "array": self._psar_vals, "type": "overlay",
             "style": "step", "color": "#ff5722"},
            {"name": f"ROC({self.roc_period})", "array": self._roc,
             "type": "subplot", "zero_line": True, "color": "#2196f3"},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi,
             "type": "subplot", "y_range": [0, 100],
             "horizontal_lines": [self.mfi_threshold], "color": "#ab47bc"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay("PSAR", datetimes, self._psar_vals, style="step", color="#ff5722"),
            ],
            "subplots": [
                self._make_subplot(
                    f"ROC({self.roc_period})",
                    [self._make_subplot_trace("ROC", datetimes, self._roc, color="#2196f3")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#ab47bc")],
                    horizontal_lines=[self.mfi_threshold], y_range=[0, 100],
                ),
            ],
        }
