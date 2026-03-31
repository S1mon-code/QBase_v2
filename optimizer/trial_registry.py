"""Append-only trial log for Deflated Sharpe Ratio calculation.

Every Optuna trial is recorded. Cannot be deleted.
Stored at research_log/trials/trial_registry.jsonl (one JSON per line).
"""

from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    """Resolve project root (parent of optimizer/ directory)."""
    return Path(__file__).resolve().parent.parent


class TrialRegistry:
    """Append-only trial log for optimization experiments.

    Uses JSONL format (one JSON object per line) for efficient appending.
    File locking via fcntl ensures thread safety.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            path = _project_root() / "research_log" / "trials" / "trial_registry.jsonl"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        strategy: str,
        params: dict[str, Any],
        sharpe: float,
        score: float,
        regime: str,
        symbol: str,
        freq: str,
        n_trades: int,
        status: str = "active",
    ) -> None:
        """Append a trial record. Thread-safe via file lock."""
        trial_id = f"trial_{self.get_total_trials() + 1:04d}"
        entry = {
            "id": trial_id,
            "strategy": strategy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "regime": regime,
            "direction": "",
            "symbol": symbol,
            "freq": freq,
            "sharpe": sharpe,
            "score": score,
            "n_trades": n_trades,
            "status": status,
        }

        with open(self._path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _read_all(self) -> list[dict[str, Any]]:
        """Read all trial records from JSONL file."""
        if not self._path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self._path) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return records

    def get_total_trials(self) -> int:
        """Count total recorded trials."""
        return len(self._read_all())

    def get_all_sharpes(self) -> list[float]:
        """Return all recorded Sharpe ratios (for DSR calculation)."""
        return [r["sharpe"] for r in self._read_all()]

    def get_trials_for_strategy(self, strategy: str) -> list[dict[str, Any]]:
        """Return all trials for a specific strategy."""
        return [r for r in self._read_all() if r["strategy"] == strategy]
