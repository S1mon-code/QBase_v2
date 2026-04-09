"""MildTrendLongI1hV37 — Fisher Transform + OBV EMA Direction.

Economic logic: The Fisher Transform compresses price into a Gaussian
distribution, making turning points sharper and earlier than raw oscillators.
Pairing with OBV trend (above/below its EMA) filters out false Fisher signals
in low-volume chop.  We capture money from traders who rely on lagging
indicators and enter trends too late.
"""
from __future__ import annotations

import numpy as np

from indicators._utils import _ema
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV37(TrendingStrategy):
    """Fisher Transform direction confirmed by OBV EMA trend.

    Signal logic:
        Fisher > trigger AND OBV > OBV_EMA → +min(1.0, |fisher|/2)
        Fisher < trigger AND OBV < OBV_EMA → -min(1.0, |fisher|/2)
        Disagree → 0.0
    """

    name = "mild_trend_long_I_1h_v37"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50  # fisher_period + obv_ema_period + 20

    # Optimizable parameters
    fisher_period: int = 10
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
        """Precompute Fisher Transform and OBV with EMA."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._fisher, self._trigger = fisher_transform(
            self._highs, self._lows, period=self.fisher_period,
        )
        raw_obv = obv(self._closes, self._volumes)
        self._obv = raw_obv
        self._obv_ema = _ema(raw_obv, self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from Fisher/trigger cross + OBV direction."""
        fisher_val = self._fisher[bar_index]
        trigger_val = self._trigger[bar_index]
        obv_val = self._obv[bar_index]
        obv_ema_val = self._obv_ema[bar_index]

        if (
            np.isnan(fisher_val) or np.isnan(trigger_val)
            or np.isnan(obv_val) or np.isnan(obv_ema_val)
        ):
            return 0.0

        fisher_bull = fisher_val > trigger_val
        obv_bull = obv_val > obv_ema_val

        strength = min(1.0, abs(fisher_val) / 2.0)

        if fisher_bull and obv_bull:
            return strength
        if not fisher_bull and not obv_bull:
            return -strength

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "fisher_transform", "params": {"period": self.fisher_period}},
            {"name": "obv_ema", "params": {"obv_ema_period": self.obv_ema_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"OBV EMA({self.obv_ema_period})",
                [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#bb86fc")],
            ),
            self._make_subplot(
                f"Fisher({self.fisher_period})",
                [
                    self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#bb86fc"),
                    self._make_subplot_trace("Trigger", datetimes, self._trigger, color="#4fc3f7"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}

