"""Regime matcher: filter labeled periods for training data selection.

Loads regime labels from ``data/regime_labels/{instrument}.yaml`` and
provides convenience functions to retrieve periods by regime, direction,
and data split.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from regime.schema import RegimeConfig, RegimeLabel, load_labels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LABELS_DIR = _PROJECT_ROOT / "data" / "regime_labels"


def _load_instrument_labels(instrument: str) -> RegimeConfig:
    """Load the label file for a given instrument.

    Args:
        instrument: Futures instrument code (e.g. 'I', 'RB').

    Returns:
        The parsed RegimeConfig.

    Raises:
        FileNotFoundError: If the label file does not exist.
    """
    path = _LABELS_DIR / f"{instrument}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No regime labels found for instrument '{instrument}' at {path}"
        )
    return load_labels(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_regime_periods(
    instrument: str,
    regime: str,
    direction: str | None = None,
    split: str | None = None,
) -> list[tuple[date, date]]:
    """Return matching regime periods as (buffer_start, buffer_end) tuples.

    Loads labels from ``data/regime_labels/{instrument}.yaml`` and filters
    by the provided regime, optional direction, and optional split.

    When a label has no buffer dates, the core (start, end) is returned
    instead.

    Args:
        instrument: Futures instrument code.
        regime: Regime type to filter (e.g. 'strong_trend').
        direction: Optional direction filter ('up', 'down', 'neutral').
        split: Optional split filter ('train', 'oos', 'holdout').

    Returns:
        Sorted list of (buffer_start, buffer_end) date tuples.

    Raises:
        FileNotFoundError: If no label file exists for the instrument.
    """
    config = _load_instrument_labels(instrument)

    periods: list[tuple[date, date]] = []
    for lbl in config.labels:
        if lbl.regime != regime:
            continue
        if direction is not None and lbl.direction != direction:
            continue
        if split is not None and lbl.split != split:
            continue

        period_start = lbl.buffer_start if lbl.buffer_start is not None else lbl.start
        period_end = lbl.buffer_end if lbl.buffer_end is not None else lbl.end
        periods.append((period_start, period_end))

    periods.sort(key=lambda p: p[0])
    return periods


def get_train_periods(
    instrument: str,
    regime: str,
    direction: str | None = None,
) -> list[tuple[date, date]]:
    """Shortcut: return regime periods assigned to the train split.

    Args:
        instrument: Futures instrument code.
        regime: Regime type to filter.
        direction: Optional direction filter.

    Returns:
        Sorted list of (buffer_start, buffer_end) date tuples.
    """
    return get_regime_periods(instrument, regime, direction, split="train")


def get_oos_periods(
    instrument: str,
    regime: str,
    direction: str | None = None,
) -> list[tuple[date, date]]:
    """Shortcut: return regime periods assigned to the oos split.

    Args:
        instrument: Futures instrument code.
        regime: Regime type to filter.
        direction: Optional direction filter.

    Returns:
        Sorted list of (buffer_start, buffer_end) date tuples.
    """
    return get_regime_periods(instrument, regime, direction, split="oos")


def get_holdout_periods(
    instrument: str,
    regime: str,
    direction: str | None = None,
) -> list[tuple[date, date]]:
    """Shortcut: return regime periods assigned to the holdout split.

    Args:
        instrument: Futures instrument code.
        regime: Regime type to filter.
        direction: Optional direction filter.

    Returns:
        Sorted list of (buffer_start, buffer_end) date tuples.
    """
    return get_regime_periods(instrument, regime, direction, split="holdout")
