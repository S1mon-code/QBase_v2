"""MildTrendLongI1hV46 — PPO + OBV Direction.

Economic logic: Percentage Price Oscillator normalizes the MACD concept into
percentage terms, making it comparable across price levels. On-Balance Volume
tracks cumulative buying/selling pressure. When PPO crosses above its signal
line and OBV is rising (above its own EMA), both price momentum and volume
accumulation agree — a strong confirmation for iron-ore trend continuation.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.ppo import ppo
from indicators.volume.obv import obv
from indicators._utils import _ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV46(TrendingStrategy):
    """PPO line vs signal confirmed by OBV direction (vs EMA).

    Signal logic:
        - PPO line > PPO signal AND OBV > EMA(OBV): +min(1.0, abs(ppo_line) / 3)
        - PPO line < PPO signal AND OBV < EMA(OBV): -min(1.0, abs(ppo_line) / 3)
        - Disagreement: 0.0
    """

    name = "long_I_1h_v46"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 66  # ppo_slow(26) + obv_ema_period(20) + 20

    ppo_fast: int = 12
    ppo_slow: int = 26
    obv_ema_period: int = 20
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
        """Precompute PPO, OBV, and OBV EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ppo_line, self._ppo_signal, _ = ppo(
            self._closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=9,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_ema = _ema(self._obv, period=self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on PPO and OBV direction."""
        ppo_val = self._ppo_line[bar_index]
        ppo_sig = self._ppo_signal[bar_index]
        obv_val = self._obv[bar_index]
        obv_ema_val = self._obv_ema[bar_index]

        if (
            np.isnan(ppo_val)
            or np.isnan(ppo_sig)
            or np.isnan(obv_val)
            or np.isnan(obv_ema_val)
        ):
            return 0.0

        strength = min(1.0, abs(ppo_val) / 3.0)

        if ppo_val > ppo_sig and obv_val > obv_ema_val:
            return strength
        if ppo_val < ppo_sig and obv_val < obv_ema_val:
            return -strength
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "ppo",
                "params": {"fast": self.ppo_fast, "slow": self.ppo_slow, "signal": 9},
            },
            {"name": "obv", "params": {}},
            {"name": "obv_ema", "params": {"period": self.obv_ema_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"OBV EMA({self.obv_ema_period})",
                [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#bb86fc")],
            ),
            self._make_subplot(
                f"PPO({self.ppo_fast},{self.ppo_slow})",
                [
                    self._make_subplot_trace("PPO", datetimes, self._ppo_line, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._ppo_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}

