"""Horizon Diversification Constraints.

Ensures each horizon category (fast/medium/slow) has at least a minimum
weight allocation in the portfolio.
"""

from __future__ import annotations


def check_horizon_balance(
    weights: dict[str, float],
    strategy_horizons: dict[str, str],
    min_per_horizon: float = 0.15,
) -> dict[str, float]:
    """Ensure each horizon has >= min_per_horizon total weight.

    If a horizon is under-weight, transfer weight from the heaviest horizon.

    Parameters
    ----------
    weights : dict[str, float]
        Strategy weights (should sum to ~1.0).
    strategy_horizons : dict[str, str]
        Mapping of strategy name to horizon: ``"fast"``, ``"medium"``, or ``"slow"``.
    min_per_horizon : float
        Minimum total weight for each horizon that has strategies.

    Returns
    -------
    dict[str, float]
        Adjusted weights with horizon balance enforced.
    """
    if not weights or not strategy_horizons:
        return dict(weights)

    # Group strategies by horizon
    horizon_groups: dict[str, list[str]] = {}
    for s, h in strategy_horizons.items():
        if s in weights:
            horizon_groups.setdefault(h, []).append(s)

    # Only balance horizons that have strategies
    active_horizons = {h for h in horizon_groups if horizon_groups[h]}
    if len(active_horizons) <= 1:
        return dict(weights)

    result = dict(weights)

    for _ in range(10):  # iterate to convergence
        # Compute horizon totals
        horizon_totals: dict[str, float] = {}
        for h, strategies in horizon_groups.items():
            horizon_totals[h] = sum(result[s] for s in strategies)

        # Find under-weight and over-weight horizons
        under: dict[str, float] = {}
        for h in active_horizons:
            deficit = min_per_horizon - horizon_totals[h]
            if deficit > 1e-10:
                under[h] = deficit

        if not under:
            break

        # Find the heaviest horizon to donate from
        heaviest = max(
            (h for h in active_horizons if h not in under),
            key=lambda h: horizon_totals[h],
            default=None,
        )
        if heaviest is None:
            break

        total_deficit = sum(under.values())
        available = horizon_totals[heaviest] - min_per_horizon
        if available <= 0:
            break

        transfer = min(total_deficit, available)

        # Scale down heaviest horizon strategies proportionally
        heavy_total = horizon_totals[heaviest]
        if heavy_total > 0:
            scale_down = transfer / heavy_total
            for s in horizon_groups[heaviest]:
                result[s] *= (1.0 - scale_down)

        # Distribute to under-weight horizons proportionally
        for h, deficit in under.items():
            share = (deficit / total_deficit) * transfer
            h_strategies = horizon_groups[h]
            h_total = horizon_totals[h]
            for s in h_strategies:
                if h_total > 0:
                    result[s] += share * (result[s] / max(h_total, 1e-10))
                else:
                    result[s] += share / len(h_strategies)

    # Normalise
    total = sum(result.values())
    if total > 0:
        result = {s: w / total for s, w in result.items()}

    return result
