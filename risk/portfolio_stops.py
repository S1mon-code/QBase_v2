"""Four-level portfolio stop-loss system.

Levels (defaults from config/settings.yaml):
  warning:        drawdown -10%  -> log + review
  reduce:         drawdown -15%  -> halve all positions
  circuit:        drawdown -20%  -> full liquidation
  daily_circuit:  daily loss -5% -> liquidate for the day
"""

from __future__ import annotations

from config import get_settings


class PortfolioStops:
    """Portfolio-level drawdown and daily loss circuit breaker.

    Parameters
    ----------
    warning : float
        Drawdown threshold for warning (negative, e.g. -0.10).
    reduce : float
        Drawdown threshold for position reduction.
    circuit : float
        Drawdown threshold for full circuit breaker.
    daily : float
        Daily P&L threshold for daily circuit breaker.
    """

    def __init__(
        self,
        warning: float | None = None,
        reduce: float | None = None,
        circuit: float | None = None,
        daily: float | None = None,
    ) -> None:
        """Initialise from explicit values or config defaults."""
        settings = get_settings()
        self._warning = warning if warning is not None else settings["stop_warning"]
        self._reduce = reduce if reduce is not None else settings["stop_reduce"]
        self._circuit = circuit if circuit is not None else settings["stop_circuit"]
        self._daily = daily if daily is not None else settings["stop_daily"]

    def check(self, current_drawdown: float, daily_pnl_pct: float) -> str:
        """Evaluate portfolio status.

        Parameters
        ----------
        current_drawdown : float
            Current drawdown as a negative fraction (e.g. -0.12).
        daily_pnl_pct : float
            Today's P&L as a fraction (e.g. -0.03).

        Returns
        -------
        str
            One of ``"normal"``, ``"warning"``, ``"reduce"``,
            ``"circuit"``, or ``"daily_circuit"``.
        """
        # Daily circuit breaker takes priority.
        if daily_pnl_pct <= self._daily:
            return "daily_circuit"

        # Drawdown levels (most severe first).
        if current_drawdown <= self._circuit:
            return "circuit"
        if current_drawdown <= self._reduce:
            return "reduce"
        if current_drawdown <= self._warning:
            return "warning"

        return "normal"

    @staticmethod
    def get_position_multiplier(level: str) -> float:
        """Return the position multiplier for a given stop level.

        Parameters
        ----------
        level : str
            Stop level string from :meth:`check`.

        Returns
        -------
        float
            Multiplier: 1.0 for normal/warning, 0.5 for reduce,
            0.0 for circuit/daily_circuit.
        """
        _multipliers = {
            "normal": 1.0,
            "warning": 1.0,
            "reduce": 0.5,
            "circuit": 0.0,
            "daily_circuit": 0.0,
        }
        return _multipliers.get(level, 1.0)
