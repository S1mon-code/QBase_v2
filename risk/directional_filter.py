"""Apply fundamental direction constraint to signals.

Reads the per-instrument directional view from config/fundamental_views.yaml
and filters raw signals:

- ``long``    -> max(0, signal)
- ``short``   -> min(0, signal)
- ``neutral`` -> signal (no constraint)
"""

from __future__ import annotations

from config import get_fundamental_views


def load_direction(instrument: str) -> str:
    """Load the fundamental directional view for *instrument*.

    Parameters
    ----------
    instrument : str
        Instrument code (e.g. ``"RB"``).

    Returns
    -------
    str
        One of ``"long"``, ``"short"``, ``"neutral"``.

    Raises
    ------
    KeyError
        If the instrument is not found in the views config.
    """
    views = get_fundamental_views()
    entry = views["views"][instrument]
    return entry["direction"]


def filter_signal(signal: float, direction: str) -> float:
    """Filter a raw signal according to fundamental direction.

    Parameters
    ----------
    signal : float
        Raw signal value (positive = long, negative = short).
    direction : str
        ``"long"``, ``"short"``, or ``"neutral"``.

    Returns
    -------
    float
        Filtered signal value.
    """
    if direction == "long":
        return max(0.0, signal)
    if direction == "short":
        return min(0.0, signal)
    return signal


class DirectionalFilter:
    """Stateful directional filter for a single instrument.

    Parameters
    ----------
    instrument : str
        Instrument code.  Direction is loaded from config on init.
    """

    def __init__(self, instrument: str) -> None:
        """Initialise and load direction from config."""
        self._instrument = instrument
        self._direction = load_direction(instrument)

    @property
    def direction(self) -> str:
        """Current directional view."""
        return self._direction

    def apply(self, signal: float) -> float:
        """Filter a signal through the directional constraint.

        Parameters
        ----------
        signal : float
            Raw signal value.

        Returns
        -------
        float
            Filtered signal.
        """
        return filter_signal(signal, self._direction)

    def reload(self) -> None:
        """Reload direction from config (e.g. after weekly update)."""
        self._direction = load_direction(self._instrument)
