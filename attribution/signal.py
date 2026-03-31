"""Layer A: Signal Attribution (Shapley + Ablation).

Decomposes strategy Sharpe ratio into per-signal contributions using either
exact Shapley values (<=4 signals) or ablation tests (>4 signals).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Set

import numpy as np


@dataclass(frozen=True)
class SignalContribution:
    """Single signal's contribution to strategy performance."""

    name: str
    contribution: float  # Shapley value or ablation delta
    pct_of_total: float  # as percentage of total Sharpe


@dataclass(frozen=True)
class SignalAttributionResult:
    """Aggregated signal attribution result."""

    method: str  # "shapley" or "ablation"
    baseline_sharpe: float
    contributions: tuple[SignalContribution, ...]
    dominant: str  # name of highest contributor
    redundant: tuple[str, ...]  # names with < 5% contribution


def _shapley_value(
    signal: str,
    all_signals: list[str],
    evaluate_fn: Callable[[Set[str]], float],
) -> float:
    """Compute exact Shapley value for a single signal.

    Shapley(i) = sum over all coalitions S not containing i:
        weight(|S|) * [v(S ∪ {i}) - v(S)]
    where weight(|S|) = |S|! * (n - |S| - 1)! / n!
    """
    others = [s for s in all_signals if s != signal]
    n = len(all_signals)
    value = 0.0

    for size in range(len(others) + 1):
        weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
        for coalition in combinations(others, size):
            coalition_set = set(coalition)
            marginal = evaluate_fn(coalition_set | {signal}) - evaluate_fn(coalition_set)
            value += weight * marginal

    return value


def shapley_attribution(
    signal_names: list[str],
    evaluate_fn: Callable[[Set[str]], float],
) -> SignalAttributionResult:
    """Compute Shapley values for each signal.

    Exact method -- evaluates all 2^N coalitions.
    Only use when len(signal_names) <= 4.

    Args:
        signal_names: List of signal identifiers.
        evaluate_fn: Callable that takes a set of active signal names
                     and returns the Sharpe ratio for that combination.

    Returns:
        SignalAttributionResult with Shapley-based contributions.
    """
    if len(signal_names) == 0:
        return SignalAttributionResult(
            method="shapley",
            baseline_sharpe=0.0,
            contributions=(),
            dominant="",
            redundant=(),
        )

    baseline_sharpe = evaluate_fn(set(signal_names))

    raw_values: dict[str, float] = {}
    for name in signal_names:
        raw_values[name] = _shapley_value(name, signal_names, evaluate_fn)

    total_abs = sum(abs(v) for v in raw_values.values())

    contributions: list[SignalContribution] = []
    for name in signal_names:
        pct = (raw_values[name] / baseline_sharpe * 100.0) if baseline_sharpe != 0.0 else 0.0
        contributions.append(
            SignalContribution(
                name=name,
                contribution=raw_values[name],
                pct_of_total=pct,
            )
        )

    contributions_tuple = tuple(sorted(contributions, key=lambda c: c.contribution, reverse=True))
    dominant = contributions_tuple[0].name if contributions_tuple else ""
    redundant = tuple(c.name for c in contributions_tuple if abs(c.pct_of_total) < 5.0)

    return SignalAttributionResult(
        method="shapley",
        baseline_sharpe=baseline_sharpe,
        contributions=contributions_tuple,
        dominant=dominant,
        redundant=redundant,
    )


def ablation_attribution(
    signal_names: list[str],
    baseline_sharpe: float,
    evaluate_fn: Callable[[str], float],
) -> SignalAttributionResult:
    """Ablation test -- disable each signal one at a time.

    Use when len(signal_names) > 4.

    Args:
        signal_names: List of signal identifiers.
        baseline_sharpe: Sharpe with all signals active.
        evaluate_fn: Callable that takes a disabled signal name and returns
                     the Sharpe ratio with that signal neutralized.

    Returns:
        SignalAttributionResult with ablation-based contributions.
    """
    if len(signal_names) == 0:
        return SignalAttributionResult(
            method="ablation",
            baseline_sharpe=baseline_sharpe,
            contributions=(),
            dominant="",
            redundant=(),
        )

    raw_deltas: dict[str, float] = {}
    for name in signal_names:
        ablated_sharpe = evaluate_fn(name)
        raw_deltas[name] = baseline_sharpe - ablated_sharpe

    contributions: list[SignalContribution] = []
    for name in signal_names:
        pct = (raw_deltas[name] / baseline_sharpe * 100.0) if baseline_sharpe != 0.0 else 0.0
        contributions.append(
            SignalContribution(
                name=name,
                contribution=raw_deltas[name],
                pct_of_total=pct,
            )
        )

    contributions_tuple = tuple(sorted(contributions, key=lambda c: c.contribution, reverse=True))
    dominant = contributions_tuple[0].name if contributions_tuple else ""
    redundant = tuple(c.name for c in contributions_tuple if abs(c.pct_of_total) < 5.0)

    return SignalAttributionResult(
        method="ablation",
        baseline_sharpe=baseline_sharpe,
        contributions=contributions_tuple,
        dominant=dominant,
        redundant=redundant,
    )


def auto_attribution(
    signal_names: list[str],
    evaluate_fn: Callable[[Set[str]], float],
    ablation_fn: Callable[[str], float] | None = None,
    baseline_sharpe: float = 0.0,
) -> SignalAttributionResult:
    """Auto-select Shapley (<=4 signals) or Ablation (>4).

    Args:
        signal_names: List of signal identifiers.
        evaluate_fn: For Shapley -- callable(active_signals: set[str]) -> float.
        ablation_fn: For Ablation -- callable(disabled_signal: str) -> float.
                     Required when len(signal_names) > 4.
        baseline_sharpe: Sharpe with all signals active (used for ablation).

    Returns:
        SignalAttributionResult using the appropriate method.

    Raises:
        ValueError: If >4 signals and ablation_fn is not provided.
    """
    if len(signal_names) <= 4:
        return shapley_attribution(signal_names, evaluate_fn)

    if ablation_fn is None:
        raise ValueError("ablation_fn is required when signal_names > 4")

    if baseline_sharpe == 0.0:
        baseline_sharpe = evaluate_fn(set(signal_names))

    return ablation_attribution(signal_names, baseline_sharpe, ablation_fn)
