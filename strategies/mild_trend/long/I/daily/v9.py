"""MildTrendLongIDailyV9 — PSAR + PPO(60,130,45) + OI_Momentum(100).

Economic logic: Parabolic SAR provides clear trend-following stops that tighten as
trends mature. PPO (percentage-based MACD) enables cross-timeframe comparison of
momentum strength. OI momentum confirms speculative conviction is growing. Signal
scales with PPO magnitude and OI expansion rate.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.ppo import ppo
from indicators.trend.psar import psar
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV9(TrendingStrategy):
    name = "mild_trend_long_I_daily_v9"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 160

    ppo_fast: int = 60
    ppo_slow: int = 130
    oi_period: int = 100
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._psar_vals, self._psar_dir = psar(self._highs, self._lows)
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(
            self._closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=45
        )
        self._oi_mom = oi_momentum(self._oi, period=self.oi_period)

    def _generate_signal(self, bar_index: int) -> float:
        pd = self._psar_dir[bar_index]
        ppo_val = self._ppo_line[bar_index]
        ppo_h = self._ppo_hist[bar_index]
        oi_val = self._oi_mom[bar_index]

        if any(np.isnan(v) for v in [pd, ppo_val, ppo_h, oi_val]):
            return 0.0

        psar_bullish = pd == 1.0
        ppo_bullish = ppo_val > 0.0 and ppo_h > 0.0
        oi_expanding = oi_val > 0.0

        if not (psar_bullish and ppo_bullish and oi_expanding):
            return 0.0

        ppo_score = min(1.0, ppo_val / 3.0) * 0.4
        oi_score = min(1.0, oi_val / 0.15) * 0.3
        return min(1.0, 0.3 + ppo_score + oi_score)

    def get_indicator_config(self):
        return [
            {"name": "PSAR", "array": self._psar_vals, "type": "overlay", "style": "step"},
            {"name": "PPO Line", "array": self._ppo_line, "type": "subplot", "panel": "PPO", "zero_line": True},
            {"name": "PPO Signal", "array": self._ppo_signal, "type": "subplot", "panel": "PPO", "style": "dash"},
            {"name": "PPO Hist", "array": self._ppo_hist, "type": "subplot", "panel": "PPO", "style": "bar",
             "color_positive": "#66bb6a", "color_negative": "#ef5350"},
            {"name": f"OI Mom({self.oi_period})", "array": self._oi_mom, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("PSAR", datetimes, self._psar_vals, style="step", color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    "PPO",
                    [
                        self._make_subplot_trace("Line", datetimes, self._ppo_line, color="#42a5f5"),
                        self._make_subplot_trace("Signal", datetimes, self._ppo_signal, color="#ff8a80", style="dash"),
                        self._make_subplot_trace("Hist", datetimes, self._ppo_hist, style="bar",
                                                 color_positive="#66bb6a", color_negative="#ef5350"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "OI Momentum",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#4fc3f7")],
                    zero_line=True,
                ),
            ],
        }
