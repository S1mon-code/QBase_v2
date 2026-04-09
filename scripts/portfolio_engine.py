"""Portfolio Engine — combine top strategies into deployable portfolio.

Pipeline:
  1. Load all strategies + SQS scores
  2. Apply kill switch
  3. Select top strategies per quadrant (diversification)
  4. Compute return correlation matrix (real OOS daily returns; fold_sharpes fallback)
  5. Apply correlation filter (reject if pairwise corr > threshold)
  6. Weight strategies (equal / risk_parity / sqs_weighted)
  7. Compute portfolio-level metrics
  8. Generate portfolio report

Complementary to QBase_v2's portfolio/ module (Carver Signal Blending).
This engine automates SQS-based selection and correlation filtering.
"""

from __future__ import annotations

import importlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure scripts/ is on path for sibling imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sqs import apply_kill_switch, scan_all_strategies

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_CAPITAL = 10_000_000  # CNY
MAX_SINGLE_INSTRUMENT_PCT = 0.30
MAX_TOTAL_MARGIN_PCT = 0.80
MAX_LOTS_PER_STRATEGY = 100

DEFAULT_CONFIG = {
    "min_sqs": 30,
    "top_n_per_quadrant": 3,
    "corr_threshold": 0.7,
    "min_instruments": 2,
    "min_freqs": 2,
    "max_per_instrument": 3,
    "weighting_method": "sqs_weighted",
}


# ---------------------------------------------------------------------------
# OOS daily returns loader (for real correlation)
# ---------------------------------------------------------------------------

def _load_oos_returns(strategy: dict) -> np.ndarray | None:
    """Load OOS daily returns for a strategy by re-running its OOS backtest.

    Args:
        strategy: dict from scan_all_strategies with keys:
            path, instrument, direction, freq, version_dir

    Returns:
        1-D numpy array of daily returns, or None if backtest fails.
    """
    try:
        from pipeline.qbase_config import PROJECT_ROOT
        from pipeline.backtest_runner import run_qbase_backtest
        from regime.schema import load_labels
    except ImportError:
        return None

    instrument = strategy["instrument"]
    direction = strategy["direction"]
    freq = strategy["freq"]
    version_dir = strategy["version_dir"]

    # Extract version string (e.g. "v5" from "v5_+42.80%")
    v_match = re.match(r"(v\d+)", version_dir)
    if not v_match:
        return None
    version = v_match.group(1)

    # 1. Load params.yaml
    research_folder = Path(strategy["path"]).parent
    params_path = research_folder / "params.yaml"
    if not params_path.exists():
        return None

    try:
        with open(params_path) as f:
            params_data = yaml.safe_load(f) or {}
    except Exception:
        return None

    best_params = params_data.get("params", {})

    # 2. Resolve strategy class
    strategy_class = _find_strategy_class(instrument, direction, version)
    if strategy_class is None:
        return None

    # 3. Load regime labels for OOS
    dir_map = {"long": "up", "short": "down"}
    label_path = PROJECT_ROOT / "data" / "regime_labels" / f"{instrument}_{direction}.yaml"
    if not label_path.exists():
        return None

    try:
        regime_config = load_labels(label_path)
    except Exception:
        return None

    af_direction = dir_map.get(direction, direction)
    oos_labels = [
        lbl for lbl in regime_config.labels
        if lbl.split == "oos" and lbl.direction == af_direction
    ]
    if not oos_labels:
        return None

    # 4. Build active_periods and date range (using buffer dates)
    oos_active = [
        {"start": str(lbl.buffer_start or lbl.start),
         "end": str(lbl.buffer_end or lbl.end)}
        for lbl in oos_labels
    ]
    oos_start = str(min(lbl.buffer_start or lbl.start for lbl in oos_labels))
    oos_end = str(max(lbl.buffer_end or lbl.end for lbl in oos_labels))

    signal_direction = direction  # "long" or "short"

    # 5. Run OOS backtest
    try:
        result = run_qbase_backtest(
            strategy_class, best_params, instrument, freq,
            start=oos_start, end=oos_end,
            direction=signal_direction,
            active_periods=oos_active,
        )
        dr = result.daily_returns
        if hasattr(dr, "values"):
            dr = dr.values
        dr = np.asarray(dr, dtype=np.float64)
        return dr if len(dr) >= 3 else None
    except Exception:
        return None


# Strategy source mapping — which module holds strategy classes
_STRATEGY_SOURCE = {"AG": "AG", "I": "I", "RB": "RB", "LC": "LC"}


def _find_strategy_class(instrument: str, direction: str, version: str) -> type | None:
    """Resolve strategy class from instrument/direction/version.

    Returns the class or None if not found.
    """
    source = _STRATEGY_SOURCE.get(instrument, instrument)
    module_name = f"strategies.candidates.{source.lower()}_{direction}_strategies"

    # Clear cached module to allow re-import
    if module_name in sys.modules:
        del sys.modules[module_name]

    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return None

    for attr_name in sorted(dir(mod)):
        obj = getattr(mod, attr_name)
        if (isinstance(obj, type)
                and hasattr(obj, "_generate_signal")
                and hasattr(obj, "direction")
                and obj.__module__ == mod.__name__):
            name = getattr(obj, "name", "")
            v_match = re.search(r"v(\d+)", name)
            if v_match and f"v{v_match.group(1)}" == version:
                return obj
    return None


def _compute_returns_correlation(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
) -> float:
    """Compute Pearson correlation between two daily return arrays.

    Aligns on the shorter length (both start from the same OOS period).
    Returns 0.0 if insufficient data or zero variance.
    """
    min_len = min(len(returns_a), len(returns_b))
    if min_len < 10:
        return 0.0

    a = returns_a[:min_len]
    b = returns_b[:min_len]

    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0

    corr_matrix = np.corrcoef(a, b)
    r = corr_matrix[0, 1]
    return float(r) if np.isfinite(r) else 0.0


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_candidates(
    survivors: list[dict],
    config: dict | None = None,
) -> list[dict]:
    """Select top strategies per quadrant with diversification rules.

    Rules:
    - SQS >= min_sqs
    - Top N per quadrant (direction_instrument)
    - Max 3 per instrument across all quadrants
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    min_sqs = cfg["min_sqs"]
    top_n = cfg["top_n_per_quadrant"]
    max_per_inst = cfg["max_per_instrument"]

    # Filter by min SQS
    eligible = [s for s in survivors if s["sqs"] >= min_sqs]

    # Group by quadrant, take top N per quadrant (already sorted by SQS)
    by_quadrant: dict[str, list[dict]] = defaultdict(list)
    for s in eligible:
        qkey = s["quadrant"]
        if len(by_quadrant[qkey]) < top_n:
            by_quadrant[qkey].append(s)

    # Flatten
    candidates = [s for group in by_quadrant.values() for s in group]

    # Enforce max per instrument (keep highest SQS)
    candidates.sort(key=lambda x: x["sqs"], reverse=True)
    inst_count: dict[str, int] = defaultdict(int)
    filtered: list[dict] = []
    for s in candidates:
        inst = s["instrument"]
        if inst_count[inst] < max_per_inst:
            filtered.append(s)
            inst_count[inst] += 1

    return filtered


# ---------------------------------------------------------------------------
# Correlation filter
# ---------------------------------------------------------------------------

def _fold_sharpe_correlation(a: list[float], b: list[float]) -> float:
    """Compute Pearson correlation between two fold_sharpe lists.

    Uses the overlapping length. Returns 0.0 if insufficient data.
    """
    min_len = min(len(a), len(b))
    if min_len < 3:
        return 0.0

    arr_a = np.array(a[:min_len], dtype=np.float64)
    arr_b = np.array(b[:min_len], dtype=np.float64)

    # Handle zero-variance
    if np.std(arr_a) < 1e-10 or np.std(arr_b) < 1e-10:
        return 0.0

    corr_matrix = np.corrcoef(arr_a, arr_b)
    r = corr_matrix[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def apply_correlation_filter(
    candidates: list[dict],
    threshold: float = 0.7,
) -> list[dict]:
    """Remove highly correlated strategies, keeping higher SQS.

    Loads real OOS daily returns for each candidate and computes pairwise
    Pearson correlation. Falls back to fold_sharpes proxy when daily returns
    cannot be loaded.
    """
    if not candidates:
        return []

    n = len(candidates)
    print(f"\n[corr] Loading OOS daily returns for {n} candidates ...")

    # Pre-load daily returns for all candidates
    returns_cache: dict[str, np.ndarray | None] = {}
    for i, c in enumerate(candidates):
        key = c["path"]
        print(f"  [{i+1}/{n}] {c.get('quadrant', '?')} / "
              f"{c.get('version_dir', '?')} / {c.get('freq', '?')} ... ",
              end="", flush=True)
        dr = _load_oos_returns(c)
        returns_cache[key] = dr
        if dr is not None:
            print(f"OK ({len(dr)} days)")
        else:
            print("FALLBACK (fold_sharpes)")

    loaded = sum(1 for v in returns_cache.values() if v is not None)
    print(f"[corr] Loaded {loaded}/{n} daily return series; "
          f"{n - loaded} will use fold_sharpes fallback\n")

    # Greedy selection: keep highest SQS first, reject if correlated
    selected: list[dict] = []

    for candidate in candidates:
        c_key = candidate["path"]
        c_returns = returns_cache.get(c_key)
        c_folds = candidate.get("fold_sharpes", [])
        is_redundant = False

        for existing in selected:
            e_key = existing["path"]
            e_returns = returns_cache.get(e_key)

            # Prefer real daily returns correlation
            if c_returns is not None and e_returns is not None:
                corr = _compute_returns_correlation(c_returns, e_returns)
            else:
                # Fallback: fold_sharpes proxy
                e_folds = existing.get("fold_sharpes", [])
                corr = _fold_sharpe_correlation(c_folds, e_folds)

            if corr > threshold:
                print(f"  [corr] REJECT {candidate.get('version_dir', '?')} "
                      f"({candidate.get('quadrant', '?')}/{candidate.get('freq', '?')}) "
                      f"— corr={corr:.3f} with "
                      f"{existing.get('version_dir', '?')} "
                      f"({existing.get('quadrant', '?')}/{existing.get('freq', '?')})")
                is_redundant = True
                break

        if not is_redundant:
            selected.append(candidate)

    return selected


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------

def equal_weight(strategies: list[dict]) -> dict[str, float]:
    """1/N weighting."""
    n = len(strategies)
    if n == 0:
        return {}
    w = 1.0 / n
    return {s["path"]: w for s in strategies}


def risk_parity(strategies: list[dict]) -> dict[str, float]:
    """Weight inversely proportional to OOS max drawdown (proxy for risk).

    Falls back to equal weight if all drawdowns are zero.
    """
    if not strategies:
        return {}

    inv_risks: list[float] = []
    for s in strategies:
        dd = max(s.get("oos_max_dd", 0.01), 0.001)
        inv_risks.append(1.0 / dd)

    total = sum(inv_risks)
    if total <= 0:
        return equal_weight(strategies)

    return {
        s["path"]: ir / total
        for s, ir in zip(strategies, inv_risks)
    }


def sqs_weighted(strategies: list[dict]) -> dict[str, float]:
    """Weight proportional to SQS score."""
    if not strategies:
        return {}

    total_sqs = sum(s["sqs"] for s in strategies)
    if total_sqs <= 0:
        return equal_weight(strategies)

    return {s["path"]: s["sqs"] / total_sqs for s in strategies}


_WEIGHTING_METHODS = {
    "equal": equal_weight,
    "risk_parity": risk_parity,
    "sqs_weighted": sqs_weighted,
}


# ---------------------------------------------------------------------------
# Capital allocation
# ---------------------------------------------------------------------------

def allocate_capital(
    strategies: list[dict],
    weights: dict[str, float],
) -> list[dict]:
    """Compute position sizing per strategy.

    Returns strategies with added 'weight' and 'lots' fields.
    """
    allocated: list[dict] = []

    for s in strategies:
        w = weights.get(s["path"], 0.0)
        # Lots: weight * max_lots, rounded to 10-lot multiples
        raw_lots = w * MAX_LOTS_PER_STRATEGY
        lots = int(round(raw_lots / 10) * 10)
        lots = min(max(lots, 0), MAX_LOTS_PER_STRATEGY)

        entry = {**s, "weight": round(w, 4), "lots": lots}
        allocated.append(entry)

    return allocated


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def compute_portfolio_metrics(
    strategies: list[dict],
) -> dict[str, Any]:
    """Compute portfolio-level expected metrics."""
    if not strategies:
        return {
            "n_strategies": 0,
            "instruments": [],
            "freqs": [],
            "expected_sharpe": 0.0,
            "expected_return": 0.0,
            "avg_sqs": 0.0,
            "total_lots": 0,
        }

    instruments = sorted(set(s["instrument"] for s in strategies))
    freqs = sorted(set(s["freq"] for s in strategies))
    directions = sorted(set(s["direction"] for s in strategies))

    weights = [s.get("weight", 1.0 / len(strategies)) for s in strategies]
    sharpes = [s.get("oos_sharpe", 0.0) for s in strategies]
    returns_ = [s.get("oos_return", 0.0) for s in strategies]

    # Weighted average Sharpe (simplified — ignores correlation benefit)
    w_sharpe = sum(w * sh for w, sh in zip(weights, sharpes))
    w_return = sum(w * r for w, r in zip(weights, returns_))
    avg_sqs = float(np.mean([s["sqs"] for s in strategies]))
    total_lots = sum(s.get("lots", 0) for s in strategies)

    return {
        "n_strategies": len(strategies),
        "instruments": instruments,
        "freqs": freqs,
        "directions": directions,
        "expected_sharpe": round(w_sharpe, 4),
        "expected_return": round(w_return, 4),
        "avg_sqs": round(avg_sqs, 2),
        "total_lots": total_lots,
    }


# ---------------------------------------------------------------------------
# Diversification check
# ---------------------------------------------------------------------------

def _check_diversification(
    strategies: list[dict],
    config: dict,
) -> list[str]:
    """Return list of warnings if diversification rules not met."""
    warnings: list[str] = []
    instruments = set(s["instrument"] for s in strategies)
    freqs = set(s["freq"] for s in strategies)

    min_inst = config.get("min_instruments", 2)
    min_freq = config.get("min_freqs", 2)

    if len(instruments) < min_inst:
        warnings.append(
            f"Only {len(instruments)} instrument(s); want >= {min_inst}"
        )
    if len(freqs) < min_freq:
        warnings.append(
            f"Only {len(freqs)} freq(s); want >= {min_freq}"
        )
    return warnings


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_portfolio(
    research_dir: Path,
    method: str = "sqs_weighted",
    config: dict | None = None,
) -> dict:
    """Full pipeline: scan -> kill -> select -> correlate -> weight -> allocate.

    Returns portfolio dict with strategies, weights, expected metrics.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    research_dir = Path(research_dir)

    # 1. Scan all strategies
    all_strategies = scan_all_strategies(research_dir)

    # 2. Apply kill switch
    survivors, killed = apply_kill_switch(all_strategies)

    # 3. Select candidates
    candidates = select_candidates(survivors, cfg)

    # 4. Correlation filter
    corr_threshold = cfg.get("corr_threshold", 0.7)
    selected = apply_correlation_filter(candidates, threshold=corr_threshold)

    # 5. Diversification check
    div_warnings = _check_diversification(selected, cfg) if selected else []

    # 6. Weight strategies
    weight_fn = _WEIGHTING_METHODS.get(method, sqs_weighted)
    weights = weight_fn(selected)

    # 7. Allocate capital
    allocated = allocate_capital(selected, weights)

    # 8. Portfolio metrics
    metrics = compute_portfolio_metrics(allocated)

    return {
        "config": cfg,
        "method": method,
        "total_scanned": len(all_strategies),
        "total_killed": len(killed),
        "total_survivors": len(survivors),
        "total_candidates": len(candidates),
        "total_selected": len(selected),
        "strategies": allocated,
        "killed_sample": killed[:10],  # first 10 for reference
        "metrics": metrics,
        "diversification_warnings": div_warnings,
    }
