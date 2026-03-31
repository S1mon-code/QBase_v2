"""Strategy Retirement Detection.

Monitors strategy health and flags strategies for observation or removal
based on rolling performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetirementCheck:
    """Result of a strategy retirement check.

    Attributes
    ----------
    strategy : str
        Strategy name.
    action : str
        One of ``"normal"``, ``"observe"``, ``"remove"``, ``"immediate_remove"``.
    reason : str
        Human-readable explanation.
    """

    strategy: str
    action: str
    reason: str


def check_retirement(
    strategy: str,
    rolling_6m_sharpe: float,
    consecutive_loss_months: int,
    rolling_12m_sharpe: float,
    current_dd: float,
    backtest_max_dd: float,
) -> RetirementCheck:
    """Check if a strategy should be retired.

    Rules (evaluated most severe first):
        - DD > 1.5x backtest max DD -> ``"immediate_remove"``
        - 12m rolling Sharpe < -0.5 -> ``"remove"``
        - 6m rolling Sharpe < 0 -> ``"observe"`` (50% weight)
        - 3+ consecutive loss months -> ``"observe"``
        - Otherwise -> ``"normal"``

    Parameters
    ----------
    strategy : str
        Strategy name.
    rolling_6m_sharpe : float
        Rolling 6-month Sharpe ratio.
    consecutive_loss_months : int
        Number of consecutive months with negative returns.
    rolling_12m_sharpe : float
        Rolling 12-month Sharpe ratio.
    current_dd : float
        Current drawdown (negative, e.g. -0.15).
    backtest_max_dd : float
        Maximum drawdown observed in backtest (negative, e.g. -0.10).

    Returns
    -------
    RetirementCheck
        Retirement decision.
    """
    # Most severe: DD > 1.5x backtest max DD
    if backtest_max_dd < 0:
        dd_threshold = backtest_max_dd * 1.5
        if current_dd < dd_threshold:
            return RetirementCheck(
                strategy=strategy,
                action="immediate_remove",
                reason=(
                    f"Current DD ({current_dd:.2%}) exceeds 1.5x backtest max DD "
                    f"({dd_threshold:.2%})."
                ),
            )

    # 12m Sharpe < -0.5: remove
    if rolling_12m_sharpe < -0.5:
        return RetirementCheck(
            strategy=strategy,
            action="remove",
            reason=f"Rolling 12m Sharpe ({rolling_12m_sharpe:.2f}) < -0.5.",
        )

    # 6m Sharpe < 0: observe
    if rolling_6m_sharpe < 0:
        return RetirementCheck(
            strategy=strategy,
            action="observe",
            reason=f"Rolling 6m Sharpe ({rolling_6m_sharpe:.2f}) < 0. Reduce weight to 50%.",
        )

    # 3+ consecutive loss months: observe
    if consecutive_loss_months >= 3:
        return RetirementCheck(
            strategy=strategy,
            action="observe",
            reason=f"{consecutive_loss_months} consecutive loss months. Reduce weight to 50%.",
        )

    return RetirementCheck(
        strategy=strategy,
        action="normal",
        reason="Strategy is performing within normal parameters.",
    )
