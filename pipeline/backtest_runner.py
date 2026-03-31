"""Connect QBase strategies to AlphaForge V7.1 backtesting engine.

Usage:
    from pipeline.backtest_runner import run_qbase_backtest, run_on_regime_periods

    result = run_qbase_backtest(TrendMediumV6, {}, symbol="I", freq="daily",
                                 start="2018-04-16", end="2021-05-12")
    print(result.sharpe)
"""

from __future__ import annotations

import sys
import numpy as np
from typing import Any

# ── AlphaForge path ───────────────────────────────────────────────────────────
_AF_PATH = "/Users/simon/Desktop/AlphaForge"
if _AF_PATH not in sys.path:
    sys.path.insert(0, _AF_PATH)

from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.engine.event_driven import EventDrivenBacktester
from alphaforge.engine.config import BacktestConfig
from alphaforge.strategy.base import TimeSeriesStrategy

# ── Singletons ────────────────────────────────────────────────────────────────
_loader: MarketDataLoader | None = None
_specs: ContractSpecManager | None = None


def _get_loader() -> MarketDataLoader:
    global _loader
    if _loader is None:
        _loader = MarketDataLoader(f"{_AF_PATH}/data/")
    return _loader


def _get_specs() -> ContractSpecManager:
    global _specs
    if _specs is None:
        _specs = ContractSpecManager()
    return _specs


# ── AlphaForge adapter ────────────────────────────────────────────────────────

class _SignalAdapter(TimeSeriesStrategy):
    """AlphaForge strategy that follows a precomputed QBase signal array.

    Converts the continuous [-1, +1] signal into discrete buy/sell/close
    orders with vol-scaled position sizing (2% equity risk per unit signal).
    """

    name = "qbase_adapter"
    warmup = 0

    def __init__(self, signals: np.ndarray, warmup_bars: int, symbol: str) -> None:
        super().__init__()
        self._signals = signals
        self._warmup_bars = warmup_bars
        self._symbol = symbol

    def on_bar(self, context) -> None:
        i = context.bar_index
        if i < self._warmup_bars or i >= len(self._signals):
            return

        signal = float(self._signals[i])
        side, lots = context.position

        if signal > 0.25:
            target_lots = self._target_lots(context, signal)
            if lots == 0:
                if target_lots > 0:
                    context.buy(target_lots)
            elif side == -1:
                context.close_short()
                if target_lots > 0:
                    context.buy(target_lots)
        elif signal < -0.25:
            target_lots = self._target_lots(context, abs(signal))
            if lots == 0:
                if target_lots > 0:
                    context.sell(target_lots)
            elif side == 1:
                context.close_long()
                if target_lots > 0:
                    context.sell(target_lots)
        else:
            # Flat signal — close any open position
            if side == 1 and lots > 0:
                context.close_long()
            elif side == -1 and lots > 0:
                context.close_short()

    def _target_lots(self, context, strength: float) -> int:
        """Risk 2% of equity per unit signal strength."""
        price = context.close_raw
        if price <= 0:
            return 0
        spec = _get_specs().get(self._symbol)
        margin_per_lot = price * spec.multiplier * spec.margin_rate
        if margin_per_lot <= 0:
            return 0
        target = int(context.equity * 0.02 * strength / margin_per_lot)
        return max(0, min(target, 30))


# ── Core runner ───────────────────────────────────────────────────────────────

def run_qbase_backtest(
    strategy_class: type,
    params: dict[str, Any],
    symbol: str,
    freq: str = "daily",
    start: str | None = None,
    end: str | None = None,
    industrial: bool = False,
) -> Any:
    """Run a QBase strategy via AlphaForge and return BacktestResult.

    Args:
        strategy_class: QBase strategy class (subclass of QBaseStrategy).
        params:         Parameter overrides dict.
        symbol:         Instrument code, e.g. "I".
        freq:           Bar frequency, e.g. "daily", "1h".
        start:          Start date string, e.g. "2018-04-16". None = all data.
        end:            End date string. None = all data.
        industrial:     Use industrial-grade config (slower, more realistic).

    Returns:
        alphaforge BacktestResult with .sharpe, .max_drawdown, etc.
    """
    loader = _get_loader()

    # Build loader kwargs
    load_kwargs: dict[str, Any] = {"freq": freq}
    if start:
        load_kwargs["start"] = start
    if end:
        load_kwargs["end"] = end

    bars = loader.load(symbol, **load_kwargs)

    closes   = bars._close
    highs    = bars._high
    lows     = bars._low
    opens    = bars._open
    volumes  = bars._volume
    datetimes = bars._datetime

    # OI: try to get from bars, fall back to zeros
    oi = getattr(bars, "_oi", None)
    if oi is None or len(oi) != len(closes):
        oi = np.zeros(len(closes), dtype=np.float64)

    # Instantiate QBase strategy with param overrides
    strategy = strategy_class()
    for k, v in params.items():
        setattr(strategy, k, v)

    # Precompute signals
    strategy.on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
    signals = strategy.generate_signals()

    # AlphaForge config
    if industrial:
        config = BacktestConfig(
            initial_capital=10_000_000,
            volume_adaptive_spread=True,
            dynamic_margin=True,
            time_varying_spread=True,
            rollover_window_bars=20,
            margin_check_mode="daily",
            margin_call_grace_bars=3,
            asymmetric_impact=True,
            detect_locked_limit=True,
        )
    else:
        config = BacktestConfig(
            initial_capital=10_000_000,
            safe_mode=True,
            suppress_order_logs=True,
        )

    # Create adapter and run
    adapter = _SignalAdapter(signals, warmup_bars=strategy.warmup, symbol=symbol)
    engine = EventDrivenBacktester(spec_manager=_get_specs(), config=config)
    return engine.run(adapter, {symbol: bars})


def run_on_regime_periods(
    strategy_class: type,
    params: dict[str, Any],
    symbol: str,
    regime_labels: list,
    split: str = "train",
    direction: str = "up",
    freq: str = "daily",
) -> list[Any]:
    """Run backtest on each matching regime period and return list of results.

    Args:
        strategy_class: QBase strategy class.
        params:         Parameter overrides.
        symbol:         Instrument, e.g. "I".
        regime_labels:  List of RegimeLabel from load_labels().
        split:          "train", "oos", or "holdout".
        direction:      "up" or "down" — filter for this direction only.
        freq:           Bar frequency.

    Returns:
        List of BacktestResult, one per matching period.
    """
    results = []
    for lbl in regime_labels:
        if lbl.split != split:
            continue
        if lbl.direction != direction:
            continue
        try:
            r = run_qbase_backtest(
                strategy_class, params, symbol, freq,
                start=str(lbl.start), end=str(lbl.end),
            )
            results.append(r)
        except Exception as e:
            print(f"  [skip] {lbl.start}→{lbl.end}: {e}")
    return results


def aggregate_results(results: list) -> dict[str, float]:
    """Aggregate multiple BacktestResult into mean metrics.

    Returns:
        Dict with mean_sharpe, mean_calmar, mean_max_dd, mean_return, n_periods.
    """
    if not results:
        return {}
    sharpes  = [r.sharpe for r in results]
    calmars  = [r.calmar for r in results]
    max_dds  = [r.max_drawdown for r in results]
    returns  = [r.annualized_return for r in results]
    return {
        "n_periods":   len(results),
        "mean_sharpe": float(np.mean(sharpes)),
        "min_sharpe":  float(np.min(sharpes)),
        "mean_calmar": float(np.mean(calmars)),
        "mean_max_dd": float(np.mean(max_dds)),
        "mean_return": float(np.mean(returns)),
    }
