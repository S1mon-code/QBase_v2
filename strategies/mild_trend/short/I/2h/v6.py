"""MildTrendShortI2hV6 — Linear Regression Below + PPO Negative + Volume Momentum Low.

Economic logic: Price below linear regression line on 2H signals deviation from
the fitted trend. PPO negative confirms percentage-based momentum is bearish.
Low volume momentum signals fading buying interest.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression
from indicators.momentum.ppo import ppo
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV6(TrendingStrategy):
    """Price below LinReg(70) + PPO(12,35,10)<0 + VolMom(40)<1."""

    name = "mild_trend_short_I_2h_v6"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    lr_period: int = 70
    ppo_fast: int = 12
    ppo_slow: int = 35
    vm_period: int = 40
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
        self._lr = linear_regression(self._closes, self.lr_period)
        self._ppo_line, self._ppo_sig, self._ppo_hist = ppo(
            self._closes, self.ppo_fast, self.ppo_slow, 10,
        )
        self._vol_mom = volume_momentum(self._volumes, self.vm_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        lr_val = self._lr[bar_index]
        ppo_val = self._ppo_line[bar_index]
        vm = self._vol_mom[bar_index]

        if any(np.isnan(v) for v in (close, lr_val, ppo_val)):
            return 0.0

        if close >= lr_val:
            return 0.0

        dist = (lr_val - close) / lr_val
        strength = min(1.0, dist * 30.0)

        signal = -(0.25 + strength * 0.3)

        if ppo_val < 0:
            ppo_str = min(1.0, abs(ppo_val) / 3.0)
            signal -= 0.2 * ppo_str

        if not np.isnan(vm) and vm < 1.0:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LinReg({self.lr_period})", "array": self._lr, "type": "overlay", "color": "#ffab40"},
            {"name": "PPO", "array": self._ppo_line, "type": "subplot", "panel": "PPO", "zero_line": True},
            {"name": "PPO Signal", "array": self._ppo_sig, "type": "subplot", "panel": "PPO", "style": "dash"},
            {"name": f"VolMom({self.vm_period})", "array": self._vol_mom, "type": "subplot", "horizontal_lines": [1.0]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"LinReg({self.lr_period})", datetimes, self._lr, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "PPO",
                    [
                        self._make_subplot_trace("PPO", datetimes, self._ppo_line, color="#bb86fc"),
                        self._make_subplot_trace("Signal", datetimes, self._ppo_sig, style="dash", color="#78909c"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"VolMom({self.vm_period})",
                    [self._make_subplot_trace("VM", datetimes, self._vol_mom, color="#26a69a")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
