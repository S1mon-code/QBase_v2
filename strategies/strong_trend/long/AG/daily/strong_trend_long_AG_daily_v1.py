"""StrongTrendLongAGDailyV1 — KAMA(150) + ADX(70) + Twiggs(100).

Economic logic: KAMA adapts to Silver's volatile macro-driven daily trends,
filtering whipsaws via Efficiency Ratio. ADX confirms trend strength while
Twiggs Money Flow validates institutional accumulation. Agreement of adaptive
trend, strength, and volume flow produces high-conviction long signals.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.trend.adx import adx
from indicators.volume.twiggs import twiggs_money_flow
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAGDailyV1(TrendingStrategy):
    """KAMA trend direction + ADX strength + Twiggs volume confirmation.

    Signal logic:
        - KAMA rising AND ADX > threshold AND Twiggs > 0 → long signal
        - Strength scales with ADX and Twiggs magnitude

    Attributes:
        kama_period:      KAMA lookback period.
        adx_period:       ADX smoothing period.
        twiggs_period:    Twiggs Money Flow period.
        adx_threshold:    Minimum ADX for trend confirmation.
        chandelier_mult:  Chandelier Exit multiplier (optimisable).
    """

    name = "strong_trend_long_AG_daily_v1"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 250

    kama_period: int = 150
    adx_period: int = 70
    twiggs_period: int = 100
    adx_threshold: float = 20.0
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
        self._kama = kama(self._closes, period=self.kama_period)
        self._adx = adx(self._highs, self._lows, self._closes, period=self.adx_period)
        self._twiggs = twiggs_money_flow(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.twiggs_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1]
        adx_val = self._adx[bar_index]
        tw = self._twiggs[bar_index]

        if np.isnan(k) or np.isnan(k_prev) or np.isnan(adx_val) or np.isnan(tw):
            return 0.0

        kama_rising = k > k_prev
        trend_strong = adx_val > self.adx_threshold
        volume_confirm = tw > 0.0

        if kama_rising and trend_strong and volume_confirm:
            strength = min(1.0, (adx_val / 50.0) * 0.5 + tw * 2.0)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": f"ADX({self.adx_period})", "array": self._adx, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [self.adx_threshold]},
            {"name": f"Twiggs({self.twiggs_period})", "array": self._twiggs, "type": "subplot",
             "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"ADX({self.adx_period})",
                    [self._make_subplot_trace("ADX", datetimes, self._adx, color="#42a5f5")],
                    y_range=[0, 100], horizontal_lines=[self.adx_threshold],
                ),
                self._make_subplot(
                    f"Twiggs({self.twiggs_period})",
                    [self._make_subplot_trace("Twiggs MF", datetimes, self._twiggs, color="#66bb6a")],
                    zero_line=True,
                ),
            ],
        }
