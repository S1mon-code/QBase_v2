"""MildTrendShortI2hV4 — KAMA Below Price + CCI Bearish + Force Index Negative.

Economic logic: Price below KAMA on 2H signals adaptive trend is bearish.
CCI below -50 confirms price trading well below its mean. Negative Force
Index validates volume-weighted selling pressure.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.momentum.cci import cci
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV4(TrendingStrategy):
    """Price below KAMA(50) + CCI(35)<-50 + ForceIndex(30)<0."""

    name = "mild_trend_short_I_2h_v4"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 65

    kama_period: int = 50
    cci_period: int = 35
    fi_period: int = 30
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
        self._kama = kama(self._closes, period=self.kama_period)
        self._cci = cci(self._highs, self._lows, self._closes, self.cci_period)
        self._fi = force_index(self._closes, self._volumes, self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        k = self._kama[bar_index]
        cci_val = self._cci[bar_index]
        fi_val = self._fi[bar_index]

        if any(np.isnan(v) for v in (close, k, cci_val)):
            return 0.0

        if close >= k:
            return 0.0

        dist = (k - close) / k
        strength = min(1.0, dist * 35.0)

        signal = -(0.25 + strength * 0.3)

        if cci_val < -50:
            cci_str = min(1.0, abs(cci_val + 50) / 150.0)
            signal -= 0.25 * cci_str

        if not np.isnan(fi_val) and fi_val < 0:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay", "color": "#ffab40"},
            {"name": f"CCI({self.cci_period})", "array": self._cci, "type": "subplot", "zero_line": True},
            {"name": f"ForceIndex({self.fi_period})", "array": self._fi, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"CCI({self.cci_period})",
                    [self._make_subplot_trace("CCI", datetimes, self._cci, color="#bb86fc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"ForceIndex({self.fi_period})",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#ef5350")],
                    zero_line=True,
                ),
            ],
        }
