"""Layer 2: Out-of-Sample (OOS) Validation.

Checks whether optimized parameters generalize to unseen data.
Flags behavioral anomalies and suspected overfitting but does
NOT hard-reject (portfolio may need hedging strategies).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OOSResult:
    """Result of OOS validation."""

    is_sharpe: float
    oos_sharpe: float
    wf_ratio: float  # oos_sharpe / is_sharpe (inf if is_sharpe == 0)
    industrial_sharpe: float | None
    industrial_decay: float | None  # 1 - industrial/basic
    flags: tuple[str, ...]  # "suspected_overfit", "behavior_anomaly", "oos_biased_high"


def validate_oos(
    is_sharpe: float,
    oos_sharpe: float,
    is_trades: int = 0,
    oos_trades: int = 0,
    is_avg_hold: float = 0.0,
    oos_avg_hold: float = 0.0,
    industrial_sharpe: float | None = None,
) -> OOSResult:
    """Compute OOS validation result with behavioral consistency checks.

    Args:
        is_sharpe: In-sample Sharpe ratio.
        oos_sharpe: Out-of-sample Sharpe ratio.
        is_trades: Number of trades in-sample.
        oos_trades: Number of trades out-of-sample.
        is_avg_hold: Average holding period in-sample.
        oos_avg_hold: Average holding period out-of-sample.
        industrial_sharpe: Sharpe under industrial execution mode (optional).

    Returns:
        OOSResult with walk-forward ratio and diagnostic flags.

    Flags:
        - "suspected_overfit": wf_ratio < 0.5
        - "behavior_anomaly": trade freq or hold time differs > 3x
        - "oos_biased_high": oos_sharpe > is_sharpe * 1.5
    """
    if is_sharpe == 0.0:
        wf_ratio = math.inf if oos_sharpe != 0.0 else 0.0
    else:
        wf_ratio = oos_sharpe / is_sharpe

    # Compute industrial decay
    industrial_decay: float | None = None
    if industrial_sharpe is not None and oos_sharpe != 0.0:
        industrial_decay = 1.0 - industrial_sharpe / oos_sharpe

    # Collect flags
    flags: list[str] = []

    if wf_ratio < 0.5:
        flags.append("suspected_overfit")

    # Behavioral consistency: trade frequency
    if is_trades > 0 and oos_trades > 0:
        trade_ratio = max(is_trades, oos_trades) / max(min(is_trades, oos_trades), 1)
        if trade_ratio > 3.0:
            flags.append("behavior_anomaly")

    # Behavioral consistency: holding period
    if is_avg_hold > 0.0 and oos_avg_hold > 0.0:
        hold_ratio = max(is_avg_hold, oos_avg_hold) / min(is_avg_hold, oos_avg_hold)
        if hold_ratio > 3.0:
            if "behavior_anomaly" not in flags:
                flags.append("behavior_anomaly")

    # OOS biased high
    if is_sharpe > 0.0 and oos_sharpe > is_sharpe * 1.5:
        flags.append("oos_biased_high")

    return OOSResult(
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        wf_ratio=wf_ratio,
        industrial_sharpe=industrial_sharpe,
        industrial_decay=industrial_decay,
        flags=tuple(flags),
    )
