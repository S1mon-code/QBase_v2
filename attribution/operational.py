"""Layer E: Operational Attribution.

Decomposes the decay between Basic (theoretical) and Industrial (realistic)
Sharpe ratios into individual cost components.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationalAttribution:
    """Result of operational cost attribution."""

    basic_sharpe: float
    industrial_sharpe: float
    total_decay: float
    components: dict[str, float]  # {"slippage": x, "spread": y, "rollover": z, ...}


def operational_attribution(
    basic_sharpe: float,
    industrial_sharpe: float,
    component_sharpes: dict[str, float] | None = None,
) -> OperationalAttribution:
    """Decompose Industrial decay into cost components.

    Total decay = basic_sharpe - industrial_sharpe.
    If component_sharpes is provided, each entry is the Sharpe after adding
    that single cost component. The delta from basic_sharpe gives that
    component's cost.

    Args:
        basic_sharpe: Sharpe ratio under theoretical (no-cost) backtest.
        industrial_sharpe: Sharpe ratio under realistic (all costs) backtest.
        component_sharpes: Optional dict mapping cost component name to the
                          Sharpe observed when only that cost is applied.
                          E.g. {"slippage": 1.8, "spread": 1.7, "rollover": 1.95}

    Returns:
        OperationalAttribution with total decay and per-component breakdown.
    """
    total_decay = basic_sharpe - industrial_sharpe

    components: dict[str, float] = {}
    if component_sharpes is not None:
        for name, sharpe_with_cost in component_sharpes.items():
            components[name] = basic_sharpe - sharpe_with_cost

    return OperationalAttribution(
        basic_sharpe=basic_sharpe,
        industrial_sharpe=industrial_sharpe,
        total_decay=total_decay,
        components=components,
    )
