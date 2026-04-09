"""MildTrendLongI2hV23 — Williams %R + Klinger Volume Oscillator.

Economic logic: Williams %R identifies overbought/oversold zones while the
Klinger Oscillator tracks volume flow accumulation/distribution. Crossover
signals from oversold/overbought zones confirmed by volume flow produce
high-quality trend entry points for iron ore's 2h timeframe.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.williams_r import williams_r
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV23(TrendingStrategy):
    """Williams %R zone crossovers confirmed by Klinger direction.

    Signal logic:
        - %R crosses above -80 AND KVO > signal: +0.8 (leaving oversold)
        - %R crosses below -20 AND KVO < signal: -0.8 (leaving overbought)
        - %R > -50 AND KVO > signal: +0.4
        - %R < -50 AND KVO < signal: -0.4
        - Else: 0.0
    """

    name = "mild_trend_long_I_2h_v23"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 68  # wr_period(14) + klinger_fast(34) + 20

    wr_period: int = 14
    klinger_fast: int = 34
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
        """Precompute Williams %R and Klinger arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._wr = williams_r(
            self._highs, self._lows, self._closes, period=self.wr_period,
        )
        kvo, sig = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=55, signal=13,
        )
        self._kvo = kvo
        self._kvo_sig = sig

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on %R zones and Klinger."""
        wr_cur = self._wr[bar_index]
        kvo = self._kvo[bar_index]
        ks = self._kvo_sig[bar_index]

        if np.isnan(wr_cur) or np.isnan(kvo) or np.isnan(ks):
            return 0.0

        if bar_index < 1:
            return 0.0
        wr_prev = self._wr[bar_index - 1]
        if np.isnan(wr_prev):
            return 0.0

        # Crosses above -80 (leaving oversold) with volume confirmation
        if wr_cur > -80 and wr_prev <= -80 and kvo > ks:
            return 0.8
        # Crosses below -20 (leaving overbought) with volume confirmation
        if wr_cur < -20 and wr_prev >= -20 and kvo < ks:
            return -0.8
        # General zone alignment
        if wr_cur > -50 and kvo > ks:
            return 0.4
        if wr_cur < -50 and kvo < ks:
            return -0.4
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "williams_r", "params": {"period": self.wr_period}},
            {"name": "klinger", "params": {"fast": self.klinger_fast, "slow": 55, "signal": 13}},
        ]
