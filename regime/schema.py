"""Regime label schema: data classes, YAML I/O, and validation.

Defines the structure for regime labels and their configuration,
plus functions to load, save, and validate regime label files.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml


# --- Valid enum values ---

VALID_REGIMES = frozenset({"strong_trend", "mild_trend", "mean_reversion", "crisis"})
VALID_DIRECTIONS = frozenset({"up", "down", "neutral"})
VALID_SPLITS = frozenset({"train", "oos", "holdout"})


# --- Data classes ---


@dataclass(frozen=True)
class RegimeLabel:
    """A single labeled regime period.

    Attributes:
        start: Start date of the core regime period.
        end: End date of the core regime period.
        regime: One of strong_trend, mild_trend, mean_reversion, crisis.
        direction: One of up, down, neutral.
        driver: Human-readable description of the fundamental driver.
        buffer_start: Start date including the buffer window.
        buffer_end: End date including the buffer window.
        split: Data split assignment: train, oos, or holdout.
    """

    start: date
    end: date
    regime: str
    direction: str
    driver: str = ""
    buffer_start: date | None = None
    buffer_end: date | None = None
    split: str = "train"


@dataclass(frozen=True)
class RegimeConfig:
    """Full regime label configuration for one instrument.

    Attributes:
        instrument: Futures instrument code (e.g. 'I', 'RB').
        version: Schema / labeling version number.
        labeled_by: Who/what produced the labels.
        labels: Ordered list of regime labels.
    """

    instrument: str
    version: int = 1
    labeled_by: str = "auto"
    labels: tuple[RegimeLabel, ...] = ()


# --- YAML helpers ---


def _date_to_str(d: date | None) -> str | None:
    """Convert a date to ISO string, or None."""
    return d.isoformat() if d is not None else None


def _str_to_date(s: str | None) -> date | None:
    """Parse an ISO date string, or return None."""
    if s is None:
        return None
    return date.fromisoformat(str(s))


def _label_to_dict(label: RegimeLabel) -> dict[str, Any]:
    """Serialize a RegimeLabel to a plain dict for YAML output."""
    d: dict[str, Any] = {
        "start": _date_to_str(label.start),
        "end": _date_to_str(label.end),
        "regime": label.regime,
        "direction": label.direction,
    }
    if label.driver:
        d["driver"] = label.driver
    if label.buffer_start is not None:
        d["buffer_start"] = _date_to_str(label.buffer_start)
    if label.buffer_end is not None:
        d["buffer_end"] = _date_to_str(label.buffer_end)
    d["split"] = label.split
    return d


def _dict_to_label(d: dict[str, Any]) -> RegimeLabel:
    """Deserialize a dict into a RegimeLabel."""
    return RegimeLabel(
        start=_str_to_date(d["start"]),
        end=_str_to_date(d["end"]),
        regime=d["regime"],
        direction=d["direction"],
        driver=d.get("driver", ""),
        buffer_start=_str_to_date(d.get("buffer_start")),
        buffer_end=_str_to_date(d.get("buffer_end")),
        split=d.get("split", "train"),
    )


# --- Public API ---


def load_labels(path: str | Path) -> RegimeConfig:
    """Load regime labels from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        A RegimeConfig populated from the file.

    Raises:
        FileNotFoundError: If the path does not exist.
        KeyError: If required fields are missing.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    labels = tuple(
        _dict_to_label(d) for d in (raw.get("labels") or [])
    )

    return RegimeConfig(
        instrument=raw["instrument"],
        version=raw.get("version", 1),
        labeled_by=raw.get("labeled_by", "auto"),
        labels=labels,
    )


def save_labels(config: RegimeConfig, path: str | Path) -> None:
    """Save regime labels to a YAML file.

    Creates parent directories if they do not exist.

    Args:
        config: The RegimeConfig to serialize.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "instrument": config.instrument,
        "version": config.version,
        "labeled_by": config.labeled_by,
        "labels": [_label_to_dict(lbl) for lbl in config.labels],
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def validate_labels(config: RegimeConfig) -> list[str]:
    """Validate a RegimeConfig and return a list of error messages.

    Checks:
    - Required fields present on every label.
    - Regime, direction, split values are valid enums.
    - start <= end for every label.
    - buffer_start <= start and end <= buffer_end when buffers exist.
    - No overlapping core periods (start..end).

    Args:
        config: The configuration to validate.

    Returns:
        List of human-readable error strings. Empty means valid.
    """
    errors: list[str] = []

    if not config.instrument:
        errors.append("instrument is required")

    for i, lbl in enumerate(config.labels):
        prefix = f"label[{i}]"

        # Required fields
        if lbl.start is None:
            errors.append(f"{prefix}: start is required")
        if lbl.end is None:
            errors.append(f"{prefix}: end is required")
        if not lbl.regime:
            errors.append(f"{prefix}: regime is required")
        if not lbl.direction:
            errors.append(f"{prefix}: direction is required")

        # Enum validation
        if lbl.regime and lbl.regime not in VALID_REGIMES:
            errors.append(
                f"{prefix}: invalid regime '{lbl.regime}', "
                f"must be one of {sorted(VALID_REGIMES)}"
            )
        if lbl.direction and lbl.direction not in VALID_DIRECTIONS:
            errors.append(
                f"{prefix}: invalid direction '{lbl.direction}', "
                f"must be one of {sorted(VALID_DIRECTIONS)}"
            )
        if lbl.split and lbl.split not in VALID_SPLITS:
            errors.append(
                f"{prefix}: invalid split '{lbl.split}', "
                f"must be one of {sorted(VALID_SPLITS)}"
            )

        # Date ordering
        if lbl.start is not None and lbl.end is not None:
            if lbl.start > lbl.end:
                errors.append(f"{prefix}: start ({lbl.start}) > end ({lbl.end})")

        # Buffer ordering
        if lbl.buffer_start is not None and lbl.start is not None:
            if lbl.buffer_start > lbl.start:
                errors.append(
                    f"{prefix}: buffer_start ({lbl.buffer_start}) > start ({lbl.start})"
                )
        if lbl.buffer_end is not None and lbl.end is not None:
            if lbl.buffer_end < lbl.end:
                errors.append(
                    f"{prefix}: buffer_end ({lbl.buffer_end}) < end ({lbl.end})"
                )

    # Overlapping core periods
    sorted_labels = sorted(
        [(i, lbl) for i, lbl in enumerate(config.labels) if lbl.start and lbl.end],
        key=lambda x: x[1].start,
    )
    for idx in range(len(sorted_labels) - 1):
        i_a, a = sorted_labels[idx]
        i_b, b = sorted_labels[idx + 1]
        if a.end >= b.start:
            errors.append(
                f"label[{i_a}] ({a.start}..{a.end}) overlaps with "
                f"label[{i_b}] ({b.start}..{b.end})"
            )

    return errors
