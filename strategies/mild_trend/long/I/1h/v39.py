"""MildTrendLongI1hV39 — EMA(8/21) Crossover + ATR Expansion Filter.

Economic logic: Dual-EMA crossovers identify trend direction changes.  The ATR
expansion filter distinguishes genuine breakouts (expanding volatility) from
whipsaws in quiet markets.  During low-volatility regimes the signal is
attenuated to 30%, avoiding the chop that destroys most trend-following
strategies.  We profit from range traders who get caught when volatility
expands and price trends away.
"""
from __future__ import annotations

import numpy as np

from indicators._utils import _ema, _sma
from indicators.volatility.atr import atr
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV39(TrendingStrategy):
    """EMA fast/slow crossover with ATR expansion/contraction filter.

    Signal logic:
        EMA_fast > EMA_slow AND ATR > ATR_SMA → +1.0
        EMA_fast < EMA_slow AND ATR > ATR_SMA → -1.0
        ATR <= ATR_SMA → signal * 0.3 (weak)
    """

    name = "mild_trend_long_I_1h_v39"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 55  # slow_ema + atr_period + 20

    # Optimizable parameters
    fast_ema: int = 8
    slow_ema: int = 21
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
        """Precompute dual EMA and ATR with its SMA."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = _ema(self._closes, self.fast_ema)
        self._ema_slow = _ema(self._closes, self.slow_ema)
        self._atr = atr(self._highs, self._lows, self._closes, period=self.atr_period)
        self._atr_sma = _sma(self._atr, 20)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from EMA crossover with ATR expansion filter."""
        ema_f = self._ema_fast[bar_index]
        ema_s = self._ema_slow[bar_index]
        atr_val = self._atr[bar_index]
        atr_sma_val = self._atr_sma[bar_index]

        if (
            np.isnan(ema_f) or np.isnan(ema_s)
            or np.isnan(atr_val) or np.isnan(atr_sma_val)
        ):
            return 0.0

        # Determine trend direction
        if ema_f > ema_s:
            raw_signal = 1.0
        elif ema_f < ema_s:
            raw_signal = -1.0
        else:
            return 0.0

        # ATR expansion filter
        if atr_val > atr_sma_val:
            return raw_signal
        return raw_signal * 0.3

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "ema_fast", "params": {"period": self.fast_ema}},
            {"name": "ema_slow", "params": {"period": self.slow_ema}},
            {"name": "atr", "params": {"period": self.atr_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"EMA({self.fast_ema})", datetimes, self._ema_fast, color="#ffab40"),
            self._make_overlay(f"EMA({self.slow_ema})", datetimes, self._ema_slow, color="#ab47bc")
        ]
        return {"overlays": overlays, "subplots": []}

