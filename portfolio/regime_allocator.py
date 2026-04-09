"""Regime-based Activation.

Loads the current fundamental view for an instrument and determines
the position multiplier based on the regime.
"""

from __future__ import annotations

from config import get_fundamental_views


def get_active_regime(instrument: str) -> dict:
    """Load the current fundamental view for an instrument.

    Parameters
    ----------
    instrument : str
        Instrument code (e.g. ``"RB"``).

    Returns
    -------
    dict
        A dict with keys ``"direction"`` and ``"regime"``.
        Direction is one of ``"long"``, ``"short"``, ``"neutral"``.
        Regime is one of ``"long"``, ``"short"``.
    """
    views = get_fundamental_views()
    instrument_view = views.get(instrument, {})
    return {
        "direction": instrument_view.get("direction", "neutral"),
        "regime": instrument_view.get("regime", "long"),
    }


def get_position_multiplier(regime: str) -> float:
    """Return the position multiplier for a given regime.

    Parameters
    ----------
    regime : str
        Current regime label (``"long"`` or ``"short"``).

    Returns
    -------
    float
        1.0 for both long and short regimes.
    """
    return 1.0
