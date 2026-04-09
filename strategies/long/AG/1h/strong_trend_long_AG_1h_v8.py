"""StrongTrendLongAG1hV8 — PSAR + PPO(8,22,7) + ChaikinOsc(8,25).

Economic logic: Parabolic SAR provides trailing stop-based trend direction
ideal for AG's 1H momentum. PPO with fast parameters catches early momentum
surges. Chaikin Oscillator measures acceleration of accumulation/distribution.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.psar import psar
from indicators.momentum.ppo import ppo
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG1hV8(TrendingStrategy):
    name = "long_AG_1h_v8"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    psar_af_step: float = 0.02
    ppo_fast: int = 8
    ppo_slow: int = 22
    chaikin_fast: int = 8
    chandelier_mult: float = 2.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._psar_val, self._psar_dir = psar(
            self._highs, self._lows,
            af_start=self.psar_af_step, af_step=self.psar_af_step, af_max=0.2,
        )
        self._ppo_line, self._ppo_signal, _ = ppo(
            self._closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=7,
        )
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.chaikin_fast, slow=25,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        ps = self._psar_val[bar_index]
        pd = self._psar_dir[bar_index]
        pp = self._ppo_line[bar_index]
        pps = self._ppo_signal[bar_index]
        ch = self._chaikin[bar_index]

        if any(np.isnan(v) for v in (close, ps, pd, pp, pps, ch)):
            return 0.0

        # PSAR direction: 1 = bullish (SAR below price)
        psar_bull = pd > 0
        ppo_bull = pp > pps
        chaikin_bull = ch > 0.0

        if psar_bull and ppo_bull and chaikin_bull:
            strength = min(1.0, abs(pp - pps) / 1.0 * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": "PSAR", "array": self._psar_val, "type": "overlay", "style": "step"},
            {"name": f"PPO({self.ppo_fast},{self.ppo_slow})", "array": self._ppo_line,
             "type": "subplot", "zero_line": True},
            {"name": f"ChaikinOsc({self.chaikin_fast},25)", "array": self._chaikin,
             "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("PSAR", datetimes, self._psar_val, style="step", color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    f"PPO({self.ppo_fast},{self.ppo_slow})",
                    [self._make_subplot_trace("PPO", datetimes, self._ppo_line, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._ppo_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"ChaikinOsc({self.chaikin_fast},25)",
                    [self._make_subplot_trace("Chaikin", datetimes, self._chaikin, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
