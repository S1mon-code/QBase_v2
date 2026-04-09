"""StrongTrendShortAGDailyV10 — PSAR Bearish + Aroon Down Dominant + OBV Declining.

Economic logic: Parabolic SAR flipping above price is a classic trend reversal
to bearish. Aroon Down > 80 and Aroon Up < 20 confirms new lows dominate.
OBV declining confirms volume supports the downside move.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.trend.psar import psar
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV10(TrendingStrategy):
    """PSAR bearish + Aroon down dominant + OBV declining.

    Signal logic:
        PSAR above price AND Aroon_down > 80 AND OBV declining -> -0.85
        PSAR above price AND Aroon_osc < -30 -> -0.45
        else -> 0.0
    """

    name = "short_AG_daily_v10"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 150

    aroon_period: int = 100
    obv_lookback: int = 120
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._psar_vals, self._psar_dir = psar(self._highs, self._lows)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period,
        )
        self._obv = obv(self._closes, self._volumes)
        # OBV SMA for trend detection
        n = len(self._obv)
        self._obv_sma = np.full(n, np.nan)
        for i in range(self.obv_lookback - 1, n):
            window = self._obv[i - self.obv_lookback + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._obv_sma[i] = np.mean(valid)

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        ps = self._psar_vals[bar_index]
        ps_d = self._psar_dir[bar_index]
        ad = self._aroon_down[bar_index]
        ao = self._aroon_osc[bar_index]
        ov = self._obv[bar_index]
        ov_sma = self._obv_sma[bar_index]

        if np.isnan(c) or np.isnan(ps) or np.isnan(ps_d):
            return 0.0

        # PSAR above price = bearish (direction == -1)
        psar_bearish = ps_d < 0 or ps > c

        if not psar_bearish:
            return 0.0

        obv_declining = (not np.isnan(ov) and not np.isnan(ov_sma) and ov < ov_sma)

        if not np.isnan(ad) and ad > 80 and obv_declining:
            return -0.85
        if not np.isnan(ao) and ao < -30:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "PSAR", "array": self._psar_vals, "type": "overlay", "style": "step"},
            {"name": "Aroon Up", "array": self._aroon_up, "panel": "Aroon"},
            {"name": "Aroon Down", "array": self._aroon_down, "panel": "Aroon"},
            {"name": "OBV", "array": self._obv},
            {"name": "OBV SMA", "array": self._obv_sma},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay("PSAR", datetimes, self._psar_vals, style="step", color="#ff7043"),
        ]
        subplots = [
            self._make_subplot(f"Aroon({self.aroon_period})", [
                self._make_subplot_trace("Aroon Up", datetimes, self._aroon_up, color="#26a69a"),
                self._make_subplot_trace("Aroon Down", datetimes, self._aroon_down, color="#ef5350"),
            ], y_range=[0, 100], horizontal_lines=[20, 80]),
            self._make_subplot("OBV", [
                self._make_subplot_trace("OBV", datetimes, self._obv, color="#42a5f5"),
                self._make_subplot_trace("OBV SMA", datetimes, self._obv_sma, color="#ff7043", style="dash"),
            ]),
        ]
        return {"overlays": overlays, "subplots": subplots}
