"""Rebalancing Logic.

Determines when portfolio weights should be recalculated based on
time elapsed and strategy changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class RebalanceDecision:
    """Result of a rebalance check.

    Attributes
    ----------
    should_rebalance : bool
        Whether rebalancing is recommended.
    reason : str
        Human-readable reason for the decision.
    days_since_last : int
        Number of days since last rebalance.
    """

    should_rebalance: bool
    reason: str
    days_since_last: int


def check_rebalance(
    last_rebalance_date: date,
    current_date: date,
    frequency: str = "monthly",
    strategy_changed: bool = False,
) -> RebalanceDecision:
    """Determine if rebalancing is needed.

    Parameters
    ----------
    last_rebalance_date : date
        Date of the last rebalance.
    current_date : date
        Current date.
    frequency : str
        Rebalance frequency: ``"weekly"`` (7 days) or ``"monthly"`` (30 days).
    strategy_changed : bool
        Whether strategies have been added or removed.

    Returns
    -------
    RebalanceDecision
        Decision with reason.

    Raises
    ------
    ValueError
        If *frequency* is not ``"weekly"`` or ``"monthly"``.
    """
    days_since = (current_date - last_rebalance_date).days

    if strategy_changed:
        return RebalanceDecision(
            should_rebalance=True,
            reason="Strategy composition changed.",
            days_since_last=days_since,
        )

    if frequency == "weekly":
        threshold = 7
    elif frequency == "monthly":
        threshold = 30
    else:
        raise ValueError(f"Unknown frequency: {frequency!r}. Expected 'weekly' or 'monthly'.")

    if days_since >= threshold:
        return RebalanceDecision(
            should_rebalance=True,
            reason=f"Scheduled {frequency} rebalance ({days_since} days elapsed).",
            days_since_last=days_since,
        )

    return RebalanceDecision(
        should_rebalance=False,
        reason=f"Not yet due ({days_since}/{threshold} days).",
        days_since_last=days_since,
    )
