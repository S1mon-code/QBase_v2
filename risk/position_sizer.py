"""Position sizing based on risk budget and stop distance.

lots = (equity * risk_pct) / (stop_distance * multiplier)
lots = min(lots, equity * max_margin_pct / (price * multiplier * margin_rate))
lots = max(1, lots)
"""

from __future__ import annotations

import math

from config import get_settings


def calc_lots(
    equity: float,
    risk_pct: float,
    stop_distance: float,
    multiplier: float,
    price: float,
    margin_rate: float,
    max_margin_pct: float = 0.30,
) -> int:
    """Calculate the number of lots to trade.

    Parameters
    ----------
    equity : float
        Current account equity.
    risk_pct : float
        Fraction of equity risked per trade (e.g. 0.02 for 2%).
    stop_distance : float
        Distance from entry to stop in price units.
    multiplier : float
        Contract multiplier (e.g. 10 for RB).
    price : float
        Current price (used for margin calculation).
    margin_rate : float
        Margin rate as a fraction (e.g. 0.12 for 12%).
    max_margin_pct : float
        Maximum fraction of equity allocated to margin (default 0.30).

    Returns
    -------
    int
        Number of lots (minimum 1).
    """
    if stop_distance <= 0 or multiplier <= 0 or equity <= 0:
        return 1

    # Risk-based sizing.
    risk_lots = (equity * risk_pct) / (stop_distance * multiplier)

    # Margin constraint.
    margin_per_lot = price * multiplier * margin_rate
    if margin_per_lot > 0:
        max_lots_by_margin = (equity * max_margin_pct) / margin_per_lot
        risk_lots = min(risk_lots, max_lots_by_margin)

    return max(1, int(math.floor(risk_lots)))


class PositionSizer:
    """Position sizing helper that reads defaults from config.

    Parameters
    ----------
    risk_pct : float or None
        Override ``risk_per_trade`` from settings.
    max_margin_pct : float or None
        Override ``max_margin_single`` from settings.
    """

    def __init__(
        self,
        risk_pct: float | None = None,
        max_margin_pct: float | None = None,
    ) -> None:
        """Initialise from config or explicit values."""
        settings = get_settings()
        self._risk_pct = risk_pct if risk_pct is not None else settings["risk_per_trade"]
        self._max_margin_pct = max_margin_pct if max_margin_pct is not None else settings["max_margin_single"]

    def size(
        self,
        equity: float,
        stop_distance: float,
        multiplier: float,
        price: float,
        margin_rate: float,
    ) -> int:
        """Calculate lots using stored risk parameters.

        Parameters
        ----------
        equity : float
            Account equity.
        stop_distance : float
            Stop distance in price units.
        multiplier : float
            Contract multiplier.
        price : float
            Current price.
        margin_rate : float
            Margin rate fraction.

        Returns
        -------
        int
            Number of lots (minimum 1).
        """
        return calc_lots(
            equity=equity,
            risk_pct=self._risk_pct,
            stop_distance=stop_distance,
            multiplier=multiplier,
            price=price,
            margin_rate=margin_rate,
            max_margin_pct=self._max_margin_pct,
        )
