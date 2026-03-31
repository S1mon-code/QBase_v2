"""Attribution analysis module -- 5-layer decomposition of strategy alpha sources.

Layer A: Signal Attribution (Shapley + Ablation)
Layer B: Horizon Attribution (TSMOM factor regression)
Layer C: Regime Attribution (per-regime trade stats)
Layer D: Baseline Decomposition (TSMOM + Carry factor regression)
Layer E: Operational Attribution (cost decomposition)

Plus: Regime Coverage, Alpha Decay Detection, Report Generation.
"""

from attribution.signal import (
    SignalContribution,
    SignalAttributionResult,
    shapley_attribution,
    ablation_attribution,
    auto_attribution,
)
from attribution.horizon import HorizonAttributionResult, horizon_attribution
from attribution.regime import RegimeStats, RegimeAttributionResult, regime_attribution
from attribution.baseline import BaselineDecomposition, decompose_baseline
from attribution.operational import OperationalAttribution, operational_attribution
from attribution.coverage import CoverageResult, regime_coverage
from attribution.decay import DecayResult, detect_alpha_decay
from attribution.report import generate_attribution_report

__all__ = [
    "SignalContribution",
    "SignalAttributionResult",
    "shapley_attribution",
    "ablation_attribution",
    "auto_attribution",
    "HorizonAttributionResult",
    "horizon_attribution",
    "RegimeStats",
    "RegimeAttributionResult",
    "regime_attribution",
    "BaselineDecomposition",
    "decompose_baseline",
    "OperationalAttribution",
    "operational_attribution",
    "CoverageResult",
    "regime_coverage",
    "DecayResult",
    "detect_alpha_decay",
    "generate_attribution_report",
]
