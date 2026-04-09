"""StrongTrendLongAGDailyV8 — Donchian(150) + Fisher(80) + Wyckoff(80).

Economic logic: Wide Donchian channel breakout captures Silver's macro-driven
daily range expansions. Fisher Transform normalizes price into Gaussian for
clean reversal/continuation detection. Wyckoff Divergence (substituted for
generic Wyckoff) detects volume-price divergences indicating accumulation.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.wyckoff_divergence import wyckoff_divergence
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAGDailyV8(TrendingStrategy):
    """Donchian breakout + Fisher momentum + Wyckoff volume divergence.

    Signal logic:
        - Price > Donchian mid AND Fisher > 0 AND Wyckoff accumulation → long
        - Strength scales with Fisher magnitude

    Attributes:
        donchian_period: Donchian Channel period.
        fisher_period:   Fisher Transform period.
        wyckoff_lookback: Wyckoff Divergence lookback.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_AG_daily_v8"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 200

    donchian_period: int = 150
    fisher_period: int = 80
    wyckoff_lookback: int = 80
    chandelier_mult: float = 4.0

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
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.donchian_period,
        )
        self._fisher, self._fisher_signal = fisher_transform(
            self._highs, self._lows, period=self.fisher_period,
        )
        self._wyckoff = wyckoff_divergence(
            self._highs, self._lows, self._closes, self._volumes,
            lookback=self.wyckoff_lookback,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_mid = self._dc_mid[bar_index]
        fish = self._fisher[bar_index]
        wyck = self._wyckoff[bar_index]

        if any(np.isnan(v) for v in (close, dc_mid, fish, wyck)):
            return 0.0

        above_mid = close > dc_mid
        fisher_bull = fish > 0.0
        wyckoff_accum = wyck > 0.0

        if above_mid and fisher_bull and wyckoff_accum:
            strength = min(1.0, abs(fish) / 3.0 * 0.6 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"Donchian Upper({self.donchian_period})",
             "array": self._dc_upper, "type": "overlay"},
            {"name": f"Donchian Mid({self.donchian_period})",
             "array": self._dc_mid, "type": "overlay"},
            {"name": f"Donchian Lower({self.donchian_period})",
             "array": self._dc_lower, "type": "overlay"},
            {"name": f"Fisher({self.fisher_period})",
             "array": self._fisher, "type": "subplot", "zero_line": True},
            {"name": f"Wyckoff({self.wyckoff_lookback})",
             "array": self._wyckoff, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"DC Upper({self.donchian_period})", datetimes, self._dc_upper, color="#66bb6a"),
                self._make_overlay(f"DC Mid({self.donchian_period})", datetimes, self._dc_mid, style="dash", color="#ffab40"),
                self._make_overlay(f"DC Lower({self.donchian_period})", datetimes, self._dc_lower, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Fisher({self.fisher_period})",
                    [self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._fisher_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"Wyckoff({self.wyckoff_lookback})",
                    [self._make_subplot_trace("Wyckoff Div", datetimes, self._wyckoff, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
