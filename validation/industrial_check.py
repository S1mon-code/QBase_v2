"""Layer 6a: Industrial Execution Decay Check.

Measures Sharpe decay when switching from basic to industrial
execution mode (realistic slippage, partial fills, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from validation.thresholds import get_thresholds


@dataclass(frozen=True)
class IndustrialResult:
    """Result of industrial decay check."""

    basic_sharpe: float
    industrial_sharpe: float
    decay_pct: float  # (basic - industrial) / basic * 100
    verdict: str  # "normal" / "acceptable" / "warning" / "unreliable"


def check_industrial_decay(
    basic_sharpe: float,
    industrial_sharpe: float,
) -> IndustrialResult:
    """Check Sharpe decay under industrial execution mode.

    Decay thresholds:
        - < 10%: "normal"
        - 10-30%: "acceptable"
        - 30-50%: "warning" (should re-optimize under industrial mode)
        - > 50%: "unreliable" (do not include in portfolio)

    Args:
        basic_sharpe: Sharpe ratio under basic execution mode.
        industrial_sharpe: Sharpe ratio under industrial execution mode.

    Returns:
        IndustrialResult with decay percentage and verdict.
    """
    if basic_sharpe == 0.0:
        decay_pct = 0.0 if industrial_sharpe == 0.0 else 100.0
    else:
        decay_pct = (basic_sharpe - industrial_sharpe) / abs(basic_sharpe) * 100.0

    cfg = get_thresholds()["industrial"]
    if decay_pct < cfg["normal_max_decay_pct"]:
        verdict = "normal"
    elif decay_pct < cfg["acceptable_max_decay_pct"]:
        verdict = "acceptable"
    elif decay_pct < cfg["warning_max_decay_pct"]:
        verdict = "warning"
    else:
        verdict = "unreliable"

    return IndustrialResult(
        basic_sharpe=basic_sharpe,
        industrial_sharpe=industrial_sharpe,
        decay_pct=decay_pct,
        verdict=verdict,
    )
