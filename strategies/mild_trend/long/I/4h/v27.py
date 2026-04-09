"""MildTrendLongI4hV27 — McGinley Dynamic + Force Index.

Economic logic: McGinley Dynamic adjusts its speed based on market velocity,
hugging price closely in trends. Force Index combines price change with volume,
measuring the power behind each move. In iron ore, large-volume directional
moves relative to McGinley indicate institutional trend commitment.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.mcginley import mcginley_dynamic
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV27(TrendingStrategy):
    """Price vs McGinley Dynamic confirmed by Force Index.

    Signal logic:
        - Close > McGinley AND FI > 0 -> +1.0
        - Close < McGinley AND FI < 0 -> -1.0
        - Price direction but no FI confirm -> 0.3
        - Otherwise -> 0.0
    """

    name = "mild_trend_long_I_4h_v27"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 53  # mcginley_period(20) + fi_period(13) + 20

    mcginley_period: int = 20
    fi_period: int = 13
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
        self._mcginley = mcginley_dynamic(self._closes, period=self.mcginley_period)
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        mg = self._mcginley[bar_index]
        fi_val = self._fi[bar_index]

        if np.isnan(close) or np.isnan(mg) or np.isnan(fi_val):
            return 0.0

        price_above = close > mg
        price_below = close < mg

        if price_above and fi_val > 0:
            return 1.0
        if price_below and fi_val < 0:
            return -1.0
        if price_above:
            return 0.3
        if price_below:
            return -0.3
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "mcginley_dynamic", "params": {"period": self.mcginley_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]
