"""MildTrendShortIDailyV8 — Donchian Lower Break + Elder Force + BSP Selling.

Economic logic: Close below the Donchian lower channel signals a breakout
to new lows over the lookback period. Negative Elder Force Index confirms
volume-weighted selling momentum. Buying/Selling Pressure ratio below 1
validates seller dominance in the mild downtrend.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.momentum.elder_force import elder_force_index
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV8(TrendingStrategy):
    """Donchian(50) close < lower + Elder Force(20) < 0 + BSP sell > buy."""

    name = "short_I_daily_v8"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 65

    # Optimizable parameters (<=5 including chandelier_mult)
    donchian_period: int = 50
    elder_period: int = 20
    bsp_period: int = 14
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
            self._highs, self._lows, self.donchian_period,
        )
        self._elder_fi = elder_force_index(self._closes, self._volumes, self.elder_period)
        self._buy_p, self._sell_p, self._pressure_ratio = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes, self.bsp_period,
        )

        # Pre-smooth raw signal
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_lower = self._dc_lower[bar_index]
        elder_val = self._elder_fi[bar_index]
        pr = self._pressure_ratio[bar_index]

        if any(np.isnan(v) for v in (dc_lower, elder_val, pr)):
            return 0.0

        # Close must be at or below Donchian lower
        if close > dc_lower:
            return 0.0

        # Elder Force must be negative
        if elder_val >= 0:
            return 0.0

        # Proximity to lower band drives base signal
        if dc_lower > 0:
            penetration = (dc_lower - close) / dc_lower
            pen_strength = min(penetration * 50.0, 1.0)
        else:
            pen_strength = 0.5
        base = -0.3 - pen_strength * 0.3  # -0.3 to -0.6

        # BSP confirmation (sell > buy means ratio < 1)
        if pr < 1.0:
            base -= 0.1

        return np.clip(base, -0.7, 0.0)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DC Upper({self.donchian_period})", "array": self._dc_upper,
             "type": "overlay", "style": "dash", "color": "#4caf50"},
            {"name": f"DC Lower({self.donchian_period})", "array": self._dc_lower,
             "type": "overlay", "style": "dash", "color": "#f44336"},
            {"name": f"Elder Force({self.elder_period})", "array": self._elder_fi,
             "type": "subplot", "zero_line": True, "color": "#2196f3"},
            {"name": "Pressure Ratio", "array": self._pressure_ratio,
             "type": "subplot", "horizontal_lines": [1.0], "color": "#ab47bc"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"DC Upper({self.donchian_period})", datetimes,
                                   self._dc_upper, style="dash", color="#4caf50"),
                self._make_overlay(f"DC Lower({self.donchian_period})", datetimes,
                                   self._dc_lower, style="dash", color="#f44336"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Elder Force({self.elder_period})",
                    [self._make_subplot_trace("EFI", datetimes, self._elder_fi, color="#2196f3")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "BSP Ratio",
                    [self._make_subplot_trace("Ratio", datetimes, self._pressure_ratio, color="#ab47bc")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
