"""StrongTrendLongAGDailyV7 — HMA(120) + PPO(70,150,50) + MFI(80).

Economic logic: Hull Moving Average with wide period captures Silver's daily
trend with minimal lag. PPO (percentage-based MACD) normalizes momentum across
AG's varying price levels. MFI (substituted for VolumeProfile) acts as volume-
weighted RSI to detect overbought/oversold with institutional flow context.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.momentum.ppo import ppo
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAGDailyV7(TrendingStrategy):
    """HMA trend + PPO momentum + MFI volume-price confirmation.

    Signal logic:
        - HMA rising AND PPO > signal AND MFI > 50 → long signal
        - Strength scales with PPO magnitude and MFI distance from 50

    Attributes:
        hma_period:      Hull Moving Average period.
        ppo_fast:        PPO fast EMA period.
        ppo_slow:        PPO slow EMA period.
        mfi_period:      Money Flow Index period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "strong_trend_long_AG_daily_v7"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 220

    hma_period: int = 120
    ppo_fast: int = 70
    ppo_slow: int = 150
    mfi_period: int = 80
    chandelier_mult: float = 3.5

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
        self._hma = hma(self._closes, period=self.hma_period)
        self._ppo_line, self._ppo_signal, _ = ppo(
            self._closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=50,
        )
        self._mfi = mfi(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.mfi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1]
        pp = self._ppo_line[bar_index]
        ps = self._ppo_signal[bar_index]
        m = self._mfi[bar_index]

        if any(np.isnan(v) for v in (h, h_prev, pp, ps, m)):
            return 0.0

        hma_rising = h > h_prev
        ppo_bull = pp > ps
        mfi_bull = m > 50.0

        if hma_rising and ppo_bull and mfi_bull:
            strength = min(1.0, abs(pp - ps) / 2.0 * 0.5 + (m - 50.0) / 50.0 * 0.5)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": f"PPO({self.ppo_fast},{self.ppo_slow})",
             "array": self._ppo_line, "type": "subplot", "zero_line": True},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"PPO({self.ppo_fast},{self.ppo_slow})",
                    [self._make_subplot_trace("PPO", datetimes, self._ppo_line, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._ppo_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#42a5f5")],
                    y_range=[0, 100], horizontal_lines=[20, 50, 80],
                ),
            ],
        }
