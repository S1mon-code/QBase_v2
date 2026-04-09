"""Shared helpers for the dev_pipeline module.

Extracted from dev_pipeline.py to keep the orchestrator focused on flow
control while utility logic lives here.
"""

from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path
from typing import Any

from pipeline.qbase_config import ALPHAFORGE_PATH, PROJECT_ROOT
from pipeline.backtest_runner import run_qbase_backtest
from regime.schema import load_labels
from strategies.baselines.tsmom_fast import TSMOMFast
from strategies.baselines.tsmom_medium import TSMOMMedium
from strategies.baselines.tsmom_slow import TSMOMSlow

# ── Direction mapping ─────────────────────────────────────────────────────────
# Regime labels use "up"/"down"; pipeline API uses "long"/"short"
_DIR_MAP = {"long": "up", "short": "down"}

_HORIZON_BASELINE = {
    "fast": TSMOMFast,
    "medium": TSMOMMedium,
    "slow": TSMOMSlow,
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_regime_labels(symbol: str):
    """Load RegimeConfig for a symbol from data/regime_labels/{symbol}.yaml."""
    label_path = PROJECT_ROOT / "data" / "regime_labels" / f"{symbol}.yaml"
    return load_labels(label_path)


def _filter_labels(regime_config, split: str, direction_api: str, regime: str | None = None):
    """Return filtered list of RegimeLabel.

    Args:
        regime_config: RegimeConfig loaded from YAML.
        split:         "train", "oos", or "holdout".
        direction_api: "long" or "short" (converted to "up"/"down" internally).
        regime:        Regime name to filter on. None = all regimes.
    """
    af_direction = _DIR_MAP.get(direction_api, direction_api)
    results = []
    for lbl in regime_config.labels:
        if lbl.split != split:
            continue
        if lbl.direction != af_direction:
            continue
        if regime is not None and lbl.regime != regime:
            continue
        results.append(lbl)
    return results


def _run_on_labels(strategy_class, params, symbol, labels, freq, industrial=False,
                   config_overrides=None):
    """Run backtest on each RegimeLabel period; return list of results."""
    results = []
    for lbl in labels:
        try:
            r = run_qbase_backtest(
                strategy_class, params, symbol, freq,
                start=str(lbl.start), end=str(lbl.end),
                industrial=industrial,
                config_overrides=config_overrides,
            )
            results.append(r)
        except Exception as e:
            print(f"  [skip] {lbl.start}→{lbl.end}: {e}")
    return results


def _weighted_mean_sharpe(results) -> float:
    """Return period-length-weighted mean Sharpe across results."""
    if not results:
        return 0.0
    sharpes = []
    weights = []
    for r in results:
        dr = r.daily_returns
        if hasattr(dr, "values"):
            dr = dr.values
        n = len(dr)
        if n > 0:
            sharpes.append(r.sharpe)
            weights.append(n)
    if not sharpes:
        return 0.0
    total = sum(weights)
    return float(sum(s * w for s, w in zip(sharpes, weights)) / total)


def _concat_daily_returns(results) -> np.ndarray:
    """Concatenate daily returns arrays from a list of BacktestResult."""
    arrays = []
    for r in results:
        dr = r.daily_returns
        if hasattr(dr, "values"):
            dr = dr.values
        dr = np.asarray(dr, dtype=np.float64)
        if len(dr) > 0:
            arrays.append(dr)
    if not arrays:
        return np.array([], dtype=np.float64)
    return np.concatenate(arrays)


def _total_n_trades(results) -> int:
    return sum(getattr(r, "n_trades", 0) for r in results)


def _total_bars(results) -> int:
    total = 0
    for r in results:
        dr = r.daily_returns
        if hasattr(dr, "values"):
            dr = dr.values
        total += len(dr)
    return total


def _profit_factor(results) -> float:
    """Compute profit factor from BacktestResult.profit_factor across all results."""
    pf_vals = []
    for r in results:
        pf = getattr(r, "profit_factor", None)
        if pf is not None and pf > 0:
            pf_vals.append(pf)
    if not pf_vals:
        return 0.0
    return float(np.mean(pf_vals))


def _save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _load_bars_for_labels(symbol: str, labels: list, freq: str):
    """Load BarArray spanning all given regime label periods.

    Returns dict {symbol: BarArray} suitable for HTMLReportGenerator bar_data,
    or None on failure. Always pass bar_data to reports so K-line charts render.
    """
    if not labels:
        return None
    try:
        from alphaforge.data.market import MarketDataLoader
        loader = MarketDataLoader(f"{ALPHAFORGE_PATH}/data/")
        start = str(min(lbl.start for lbl in labels))
        end   = str(max(lbl.end   for lbl in labels))
        bars = loader.load(symbol, freq=freq, start=start, end=end)
        return {symbol: bars}
    except Exception as e:
        print(f"  [report] bar_data load failed ({e}); K-line will be omitted")
        return None


def _generate_html_report(result, path: Path, bar_data=None, freq: str = "daily") -> None:
    """Generate single-strategy HTML report with K-line chart.

    bar_data must be passed as {symbol: BarArray} — without it the K-line chart
    is blank. Use _load_bars_for_labels() to build bar_data before calling this.
    """
    try:
        from alphaforge.report import HTMLReportGenerator
        path.parent.mkdir(parents=True, exist_ok=True)
        reporter = HTMLReportGenerator()
        reporter.generate(result, str(path), bar_data=bar_data, freq=freq)
    except Exception as e:
        print(f"  [report] HTML generation skipped: {e}")


def _get_trade_regimes(result, labels) -> tuple[np.ndarray, np.ndarray]:
    """Tag each trade in result.trades with its regime label."""
    try:
        trades = result.trades
        if trades is None or len(trades) == 0:
            return np.array([]), np.array([])

        import pandas as pd

        # Get trade entry dates
        if "entry_time" in trades.columns:
            entry_times = pd.to_datetime(trades["entry_time"])
        elif "date" in trades.columns:
            entry_times = pd.to_datetime(trades["date"])
        else:
            return np.array([]), np.array([])

        pnls = trades["net_pnl"].values if hasattr(trades["net_pnl"], "values") else np.asarray(trades["net_pnl"])

        regime_tags = []
        for et in entry_times:
            tag = "unknown"
            for lbl in labels:
                start = pd.Timestamp(lbl.start)
                end = pd.Timestamp(lbl.end)
                if start <= et <= end:
                    tag = lbl.regime
                    break
            regime_tags.append(tag)

        return np.asarray(pnls), np.asarray(regime_tags)
    except Exception:
        return np.array([]), np.array([])
