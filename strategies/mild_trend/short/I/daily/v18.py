"""MildTrendShortIDailyV18 — HMA slope negative + RSI weak + Volume spike.

Economic logic: HMA(60) declining slope on daily Iron Ore captures smooth
bearish momentum. RSI(20) < 45 confirms fading upward pressure. Volume
spikes during the decline indicate panic selling or stop-loss cascades,
reinforcing the short conviction.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.trend.ema import ema
from indicators.momentum.rsi import rsi
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV18(TrendingStrategy):
    """HMA(60) slope < 0 + RSI(20) < 45 + Volume spike."""

    name = "mild_trend_short_I_daily_v18"
    horizon = "slow"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    hma_period: int = 60
    rsi_period: int = 20
    vol_spike_period: int = 30
    chandelier_mult: float = 3.6

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
        self._hma = hma(self._closes, self.hma_period)
        self._rsi = rsi(self._closes, self.rsi_period)
        self._vol_spike = volume_spike(self._volumes, self.vol_spike_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1]
        r = self._rsi[bar_index]
        vs = self._vol_spike[bar_index]

        if any(np.isnan(v) for v in (h, h_prev, r, vs)):
            return 0.0

        slope = h - h_prev
        if slope >= 0:
            return 0.0

        # Base from HMA slope
        slope_pct = abs(slope) / h_prev if h_prev != 0 else 0.0
        signal = -(0.3 + min(0.15, slope_pct * 200))

        # RSI weakness
        if r < 45:
            signal -= min(0.15, (45 - r) / 100.0)

        # Volume spike boost
        if vs > 1.5:
            signal -= min(0.1, (vs - 1.5) * 0.05)

        return max(-0.7, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay", "color": "#ff7043"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 45, 70]},
            {"name": "Vol Spike", "array": self._vol_spike, "type": "subplot",
             "horizontal_lines": [1.5]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    f"RSI({self.rsi_period})",
                    [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc")],
                    horizontal_lines=[30, 45, 70], y_range=[0, 100],
                ),
                self._make_subplot(
                    "Volume Spike",
                    [self._make_subplot_trace("Spike", datetimes, self._vol_spike, color="#26c6da")],
                    horizontal_lines=[1.5],
                ),
            ],
        }
