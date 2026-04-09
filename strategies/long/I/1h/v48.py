"""MildTrendLongI1hV48 — ROC + Klinger + ATR Filter.

Economic logic: Rate of Change measures raw price momentum. The Klinger Volume
Oscillator tracks volume flow relative to trend direction, providing a
volume-based confirmation. ATR expansion (current ATR > SMA of ATR) acts as a
volatility filter — expanding volatility typically accompanies genuine breakouts
in iron ore. When ATR is contracting, signals are dampened to avoid false
entries in quiet, range-bound periods.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr
from indicators._utils import _sma
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV48(TrendingStrategy):
    """ROC direction + Klinger confirmation + ATR expansion filter.

    Signal logic:
        - ROC > 0 AND KVO > signal AND ATR expanding: +min(1.0, abs(ROC) / 5)
        - ROC < 0 AND KVO < signal AND ATR expanding: -min(1.0, abs(ROC) / 5)
        - ATR not expanding: signal * 0.3
        - All disagree: 0.0
    """

    name = "long_I_1h_v48"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 68  # klinger_fast(34) + atr_period(14) + 20

    roc_period: int = 14
    klinger_fast: int = 34
    atr_period: int = 14
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
        """Precompute ROC, Klinger, ATR, and ATR SMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._kvo, self._kvo_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=55, signal=13,
        )
        self._atr = atr(
            self._highs, self._lows, self._closes, period=self.atr_period,
        )
        self._atr_sma = _sma(self._atr, period=20)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on ROC, Klinger, and ATR filter."""
        roc_val = self._roc[bar_index]
        kvo_val = self._kvo[bar_index]
        kvo_sig = self._kvo_signal[bar_index]
        atr_val = self._atr[bar_index]
        atr_sma_val = self._atr_sma[bar_index]

        if (
            np.isnan(roc_val)
            or np.isnan(kvo_val)
            or np.isnan(kvo_sig)
            or np.isnan(atr_val)
            or np.isnan(atr_sma_val)
        ):
            return 0.0

        atr_expanding = atr_val > atr_sma_val
        strength = min(1.0, abs(roc_val) / 5.0)

        if roc_val > 0 and kvo_val > kvo_sig:
            return strength if atr_expanding else strength * 0.3
        if roc_val < 0 and kvo_val < kvo_sig:
            return (-strength) if atr_expanding else (-strength * 0.3)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "roc", "params": {"period": self.roc_period}},
            {
                "name": "klinger",
                "params": {"fast": self.klinger_fast, "slow": 55, "signal": 13},
            },
            {"name": "atr", "params": {"period": self.atr_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"Klinger({self.klinger_fast})",
                [
                    self._make_subplot_trace("KVO", datetimes, self._kvo, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._kvo_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            ),
            self._make_subplot(
                f"ROC({self.roc_period})",
                [self._make_subplot_trace("ROC", datetimes, self._roc, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}

