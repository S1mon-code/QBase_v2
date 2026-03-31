"""Attribution Report Generation.

Assembles results from all attribution layers into a structured markdown report.
"""

from __future__ import annotations

from attribution.signal import SignalAttributionResult
from attribution.horizon import HorizonAttributionResult
from attribution.regime import RegimeAttributionResult
from attribution.baseline import BaselineDecomposition
from attribution.operational import OperationalAttribution


def generate_attribution_report(
    signal_result: SignalAttributionResult | None = None,
    horizon_result: HorizonAttributionResult | None = None,
    regime_result: RegimeAttributionResult | None = None,
    baseline_result: BaselineDecomposition | None = None,
    operational_result: OperationalAttribution | None = None,
    strategy_name: str = "",
    symbol: str = "",
) -> str:
    """Generate markdown attribution report.

    Assembles available attribution results into a structured report.
    Sections are included only when corresponding results are provided.

    Args:
        signal_result: Layer A result (Shapley/Ablation).
        horizon_result: Layer B result (TSMOM horizon regression).
        regime_result: Layer C result (per-regime stats).
        baseline_result: Layer D result (TSMOM + Carry decomposition).
        operational_result: Layer E result (cost decomposition).
        strategy_name: Strategy identifier for report title.
        symbol: Instrument symbol for report title.

    Returns:
        Markdown-formatted attribution report string.
    """
    lines: list[str] = []

    # Title
    title_parts = [p for p in [strategy_name, symbol] if p]
    title = " ".join(title_parts) if title_parts else "Strategy"
    lines.append(f"# {title} Attribution Report")
    lines.append("")

    # Layer A: Signal Attribution
    if signal_result is not None:
        lines.append("## Signal Attribution")
        lines.append(f"- method: {signal_result.method}")
        lines.append(f"- baseline_sharpe: {signal_result.baseline_sharpe:.3f}")
        if signal_result.contributions:
            lines.append(f"- dominant: {signal_result.dominant} "
                         f"({_find_pct(signal_result, signal_result.dominant):.1f}%)")
            for c in signal_result.contributions:
                lines.append(f"  - {c.name}: {c.contribution:.4f} ({c.pct_of_total:.1f}%)")
            if signal_result.redundant:
                lines.append(f"- redundant: {', '.join(signal_result.redundant)}")
        lines.append("")

    # Layer B: Horizon Attribution
    if horizon_result is not None:
        fp = horizon_result.horizon_fingerprint
        lines.append("## Horizon Fingerprint")
        lines.append(f"- Fast: {fp.get('fast', 0):.1f}% | "
                     f"Medium: {fp.get('medium', 0):.1f}% | "
                     f"Slow: {fp.get('slow', 0):.1f}%")
        lines.append(f"- Independent Alpha: {horizon_result.independent_alpha:.4f}")
        lines.append(f"- R²: {horizon_result.r_squared:.3f}")
        lines.append("")

    # Layer C: Regime Attribution
    if regime_result is not None:
        lines.append("## Regime Attribution")
        best_stat = _find_regime_stat(regime_result, regime_result.best_regime)
        worst_stat = _find_regime_stat(regime_result, regime_result.worst_regime)
        if best_stat:
            lines.append(f"- best: {best_stat.regime} "
                         f"({best_stat.win_rate:.0%} WR, {best_stat.total_pnl:+.2f})")
        if worst_stat:
            lines.append(f"- worst: {worst_stat.regime} "
                         f"({worst_stat.win_rate:.0%} WR, {worst_stat.total_pnl:+.2f})")
        if regime_result.regime_dependent:
            lines.append("- WARNING: strategy is regime-dependent (>30pp WR spread)")
        lines.append("")

    # Layer D: Baseline Decomposition
    if baseline_result is not None:
        lines.append("## Baseline Decomposition")
        lines.append(f"- TSMOM Beta: {baseline_result.tsmom_pct:.1f}% of return")
        lines.append(f"- Carry Beta: {baseline_result.carry_pct:.1f}% of return")
        lines.append(f"- Independent Alpha: {baseline_result.alpha_pct:.1f}% of return")
        lines.append(f"- R²: {baseline_result.r_squared:.3f}")
        lines.append("")

    # Layer E: Operational
    if operational_result is not None:
        lines.append("## Operational")
        decay_pct = (
            operational_result.total_decay / operational_result.basic_sharpe * 100.0
            if operational_result.basic_sharpe != 0.0
            else 0.0
        )
        lines.append(f"- Industrial decay: {decay_pct:.1f}%")
        lines.append(f"- Basic Sharpe: {operational_result.basic_sharpe:.3f}")
        lines.append(f"- Industrial Sharpe: {operational_result.industrial_sharpe:.3f}")
        if operational_result.components:
            lines.append("- Components:")
            for name, cost in operational_result.components.items():
                lines.append(f"  - {name}: {cost:.4f}")
        lines.append("")

    # Recommendations
    recommendations = _generate_recommendations(
        signal_result, horizon_result, regime_result, baseline_result, operational_result
    )
    if recommendations:
        lines.append("## Recommendations")
        for rec in recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)


def _find_pct(result: SignalAttributionResult, name: str) -> float:
    """Find the pct_of_total for a given signal name."""
    for c in result.contributions:
        if c.name == name:
            return c.pct_of_total
    return 0.0


def _find_regime_stat(result: RegimeAttributionResult, regime: str):
    """Find RegimeStats for a given regime name."""
    for s in result.stats:
        if s.regime == regime:
            return s
    return None


def _generate_recommendations(
    signal_result: SignalAttributionResult | None,
    horizon_result: HorizonAttributionResult | None,
    regime_result: RegimeAttributionResult | None,
    baseline_result: BaselineDecomposition | None,
    operational_result: OperationalAttribution | None,
) -> list[str]:
    """Generate actionable recommendations based on attribution results."""
    recs: list[str] = []

    if signal_result is not None and signal_result.redundant:
        recs.append(
            f"Consider removing redundant signals (<5%): {', '.join(signal_result.redundant)}"
        )

    if baseline_result is not None and abs(baseline_result.independent_alpha) < 0.02:
        recs.append(
            "Independent alpha < 2% annualized -- consider reducing portfolio weight"
        )

    if regime_result is not None and regime_result.regime_dependent:
        recs.append(
            f"Strategy regime-dependent: worst regime = {regime_result.worst_regime}"
        )

    if operational_result is not None and operational_result.basic_sharpe > 0:
        decay_pct = operational_result.total_decay / operational_result.basic_sharpe
        if decay_pct > 0.5:
            recs.append(
                f"Industrial decay > 50% ({decay_pct:.0%}) -- strategy may not survive costs"
            )

    return recs
