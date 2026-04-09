"""StrongTrendShortAGDailyV6 — HMA Bearish + PPO Negative + Force Index Negative.

Economic logic: HMA (Hull MA) is fast-reacting — price below HMA catches trend
changes early. PPO below zero confirms percentage-based momentum is bearish.
Force Index negative confirms selling pressure is dominant.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.ppo import ppo
from indicators.trend.hma import hma
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV6(TrendingStrategy):
    """HMA bearish + PPO negative + Force Index sell pressure.

    Signal logic:
        close < HMA AND PPO_line < 0 AND ForceIndex < 0 -> -0.85
        close < HMA AND PPO_line < 0 -> -0.45
        else -> 0.0
    """

    name = "strong_trend_short_AG_daily_v6"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 200

    hma_period: int = 120
    fi_period: int = 70
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, self.hma_period)
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(
            self._closes, fast=60, slow=140, signal=45,
        )
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        h = self._hma[bar_index]
        p = self._ppo_line[bar_index]
        fi = self._fi[bar_index]

        if np.isnan(c) or np.isnan(h) or np.isnan(p):
            return 0.0

        if c >= h:
            return 0.0

        if p < 0 and (not np.isnan(fi) and fi < 0):
            return -0.85
        if p < 0:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": "PPO Line", "array": self._ppo_line, "panel": "PPO"},
            {"name": "PPO Signal", "array": self._ppo_signal, "panel": "PPO"},
            {"name": f"ForceIndex({self.fi_period})", "array": self._fi, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot("PPO", [
                self._make_subplot_trace("PPO Line", datetimes, self._ppo_line, color="#42a5f5"),
                self._make_subplot_trace("PPO Signal", datetimes, self._ppo_signal, color="#ff7043"),
            ], zero_line=True),
            self._make_subplot(f"ForceIndex({self.fi_period})", [
                self._make_subplot_trace("ForceIndex", datetimes, self._fi, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
