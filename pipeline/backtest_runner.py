"""Connect QBase strategies to AlphaForge V7.2 backtesting engine.

Usage:
    from pipeline.backtest_runner import run_qbase_backtest, run_on_regime_periods

    result = run_qbase_backtest(TrendMediumV6, {}, symbol="I", freq="daily",
                                 start="2018-04-16", end="2021-05-12")
    print(result.sharpe)
"""

from __future__ import annotations

import numpy as np
from typing import Any

from pipeline.qbase_config import ALPHAFORGE_PATH, DATA_DIR

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
        _loader = MarketDataLoader(DATA_DIR)
    return _loader


def _get_specs() -> ContractSpecManager:
    global _specs
    if _specs is None:
        _specs = ContractSpecManager()
    return _specs


# ── AlphaForge adapter ────────────────────────────────────────────────────────

class _SignalAdapter(TimeSeriesStrategy):
    """AlphaForge strategy that follows a precomputed QBase signal array.

    Uses vol-targeting position sizing: scales the position so that the
    portfolio's annualized volatility equals ``TARGET_VOL * |signal|``.

    Sizing is applied on ENTRY only — the position is held at the entry
    size until the signal flips direction or enters the dead-zone.  No
    intra-hold rebalancing (daily vol noise would cause excessive trading).

    Formula:
        target_lots = TARGET_VOL * |signal| * equity
                      / (multiplier * price * realized_ann_vol)

    Caps:
        - Max margin utilization: 80% of equity.
        - Min vol lookback: 5 bars (below this → skip).
    """

    warmup = 0

    # ── Tuning knobs ─────────────────────────────────────────────────────────
    TARGET_VOL: float = 0.15           # 15% annualized portfolio vol target
    VOL_LOOKBACK: int = 20             # bars for realized vol estimate
    MAX_MARGIN_UTIL: float = 0.80      # cap: 80% equity as margin
    SIGNAL_THRESHOLD: float = 0.05     # dead-zone: |signal| ≤ this → flat
    REBALANCE_BUFFER: float = 0.10     # only rebalance if |actual - target| > 10%

    def __init__(
        self,
        signals: np.ndarray,
        warmup_bars: int,
        symbol: str,
        strategy_name: str = "qbase_adapter",
        freq: str = "daily",
    ) -> None:
        super().__init__()
        self.name = strategy_name
        self._signals = signals
        self._warmup_bars = warmup_bars
        self._symbol = symbol
        self._freq = freq
        # Intraday frequencies use fixed entry sizing (no continuous rebalance)
        self._fixed_entry = freq in ("1h", "2h", "4h", "30min", "10min", "5min")

    # ── Bar handler ──────────────────────────────────────────────────────────

    def on_bar(self, context) -> None:
        """Position management with two modes:

        Daily/4h: Continuous forecast-proportional sizing with 10% inertia buffer.
        1h and faster: Fixed entry sizing — hold until signal flips or enters dead zone.
        """
        i = context.bar_index
        if i < self._warmup_bars or i >= len(self._signals):
            return

        signal = float(self._signals[i])
        side, lots = context.position

        if self._fixed_entry:
            self._on_bar_fixed_entry(context, signal, side, lots)
        else:
            self._on_bar_continuous(context, signal, side, lots)

    def _on_bar_fixed_entry(self, context, signal, side, lots):
        """Fixed entry sizing for intraday (1h and faster).

        - Enter: vol-target sizing once
        - Hold: no rebalance regardless of signal changes
        - Exit: signal enters dead zone or direction flips
        """
        if signal > self.SIGNAL_THRESHOLD:
            if lots == 0:
                target = self._vol_target_lots(context, signal)
                if target > 0:
                    context.buy(target)
            elif side == -1:
                context.close_short()
                target = self._vol_target_lots(context, signal)
                if target > 0:
                    context.buy(target)
            # side == 1 → already long, hold

        elif signal < -self.SIGNAL_THRESHOLD:
            if lots == 0:
                target = self._vol_target_lots(context, abs(signal))
                if target > 0:
                    context.sell(target)
            elif side == 1:
                context.close_long()
                target = self._vol_target_lots(context, abs(signal))
                if target > 0:
                    context.sell(target)
            # side == -1 → already short, hold

        else:
            # Dead zone → flatten
            if side == 1 and lots > 0:
                context.close_long()
            elif side == -1 and lots > 0:
                context.close_short()

    def _on_bar_continuous(self, context, signal, side, lots):
        """Continuous forecast-proportional sizing for daily/4h (Carver method).

        Every bar: compute target, rebalance if deviation > 10%.
        """
        current_pos = lots if side == 1 else (-lots if side == -1 else 0)

        if abs(signal) <= self.SIGNAL_THRESHOLD:
            target_pos = 0
        elif signal > 0:
            target_pos = self._vol_target_lots(context, signal)
        else:
            target_pos = -self._vol_target_lots(context, abs(signal))

        # Flatten
        if target_pos == 0 and current_pos != 0:
            if side == 1:
                context.close_long()
            elif side == -1:
                context.close_short()
            return

        # Enter
        if current_pos == 0 and target_pos != 0:
            if target_pos > 0:
                context.buy(target_pos)
            else:
                context.sell(abs(target_pos))
            return

        if current_pos != 0 and target_pos != 0:
            # Direction flip
            if (current_pos > 0) != (target_pos > 0):
                if side == 1:
                    context.close_long()
                elif side == -1:
                    context.close_short()
                if target_pos > 0:
                    context.buy(target_pos)
                else:
                    context.sell(abs(target_pos))
                return

            # Same direction — rebalance if deviation > buffer
            deviation = abs(current_pos - target_pos) / max(abs(current_pos), 1)
            if deviation > self.REBALANCE_BUFFER:
                diff = target_pos - current_pos
                if diff > 0:
                    context.buy(abs(diff))
                elif diff < 0:
                    if side == 1:
                        context.close_long(lots=abs(diff))
                    elif side == -1:
                        context.close_short(lots=abs(diff))

    # ── Vol-targeting position sizer ─────────────────────────────────────────

    def _vol_target_lots(self, context, strength: float) -> int:
        """Compute lot size to achieve TARGET_VOL portfolio volatility.

        lots = TARGET_VOL * strength * equity / (multiplier * price * ann_vol)
        """
        price = context.close_raw
        if price <= 0:
            return 0

        spec = _get_specs().get(self._symbol)

        # Realized vol from recent closes
        n = min(self.VOL_LOOKBACK, context.bar_index)
        if n < 5:
            return 0
        closes = context.get_close_array(n + 1)
        rets = np.diff(closes) / closes[:-1]
        daily_vol = float(np.std(rets))
        if daily_vol <= 1e-8:
            return 0

        ann_vol = daily_vol * np.sqrt(252)

        target = int(
            self.TARGET_VOL * strength * context.equity
            / (spec.multiplier * price * ann_vol)
        )

        # Margin cap
        margin_per_lot = price * spec.multiplier * spec.margin_rate
        max_lots = (
            int(context.equity * self.MAX_MARGIN_UTIL / margin_per_lot)
            if margin_per_lot > 0
            else 0
        )

        return max(0, min(target, max_lots))


# ── Indicator panel packaging ─────────────────────────────────────────────────

# Default colors for auto-assignment (TradingView-inspired palette)
_OVERLAY_COLORS = ["#ffab40", "#ab47bc", "#26c6da", "#66bb6a", "#ef5350"]
_SUBPLOT_COLORS = ["#4fc3f7", "#ff9800", "#bb86fc", "#26a69a", "#ef5350"]


def _inject_indicator_panels(
    result,
    strategy,
    signals: np.ndarray,
    datetimes: np.ndarray,
) -> None:
    """Package strategy indicator arrays into result.metadata for AlphaForge report.

    Calls ``strategy.get_indicator_panels(datetimes)`` and appends a "Signal"
    subplot as the last panel.  If the strategy returns empty panels, only the
    signal subplot is added.
    """
    if not hasattr(result, "metadata") or result.metadata is None:
        result.metadata = {}

    panels = strategy.get_indicator_panels(datetimes)
    if not isinstance(panels, dict):
        panels = {"overlays": [], "subplots": []}

    overlays = list(panels.get("overlays", []))
    subplots = list(panels.get("subplots", []))

    # Auto-assign colors to overlays missing explicit color
    for i, ov in enumerate(overlays):
        if "color" not in ov:
            ov["color"] = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]

    # Auto-assign colors to subplot traces missing explicit color
    color_idx = 0
    for sp in subplots:
        for tr in sp.get("traces", []):
            if "color" not in tr and "color_positive" not in tr:
                tr["color"] = _SUBPLOT_COLORS[color_idx % len(_SUBPLOT_COLORS)]
                color_idx += 1

    # Append strategy signal as the last subplot
    from strategies.templates.base_strategy import QBaseStrategy

    signal_trace = QBaseStrategy._make_subplot_trace(
        name="Strategy Signal",
        datetimes=datetimes,
        data=signals,
        style="area",
        color="#4fc3f7",
    )
    signal_panel = QBaseStrategy._make_subplot(
        name="Signal",
        traces=[signal_trace],
        height_ratio=0.10,
        y_range=[-1, 1],
        zero_line=True,
    )
    subplots.append(signal_panel)

    result.metadata["indicator_panels"] = {
        "overlays": overlays,
        "subplots": subplots,
    }


# ── Core runner ───────────────────────────────────────────────────────────────

def run_qbase_backtest(
    strategy_class: type,
    params: dict[str, Any],
    symbol: str,
    freq: str = "daily",
    start: str | None = None,
    end: str | None = None,
    industrial: bool = False,
    config_overrides: dict | None = None,
    direction: str | None = None,
    active_periods: list[dict[str, str]] | None = None,
) -> Any:
    """Run a QBase strategy via AlphaForge and return BacktestResult.

    Args:
        strategy_class:  QBase strategy class (subclass of QBaseStrategy).
        params:          Parameter overrides dict.
        symbol:          Instrument code, e.g. "I".
        freq:            Bar frequency, e.g. "daily", "1h".
        start:           Start date string, e.g. "2018-04-16". None = all data.
        end:             End date string. None = all data.
        industrial:      Use industrial-grade config (slower, more realistic).
        config_overrides: Optional dict of BacktestConfig attribute overrides to
                         apply after building the config object. Example:
                         {"slippage_ticks": 2.0} sets config.slippage_ticks = 2.0.
        direction:       Directional filter: "long" (max(0,signal)),
                         "short" (min(0,signal)), or None (no filter).
        active_periods:  List of {"start": str, "end": str} dicts. When provided,
                         signals outside these periods are zeroed out — the strategy
                         only trades during active regime windows. The full K-line
                         is still visible in reports.

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

    # Apply directional filter (Layer 4)
    if direction == "long":
        signals = np.maximum(0.0, signals)
    elif direction == "short":
        signals = np.minimum(0.0, signals)

    # Apply active_periods mask — zero out signals outside regime windows
    if active_periods is not None and len(active_periods) > 0:
        import pandas as pd
        dt_series = pd.to_datetime(datetimes)
        mask = np.zeros(len(signals), dtype=bool)
        for ap in active_periods:
            ap_start = pd.Timestamp(ap["start"])
            ap_end = pd.Timestamp(ap["end"])
            mask |= (dt_series >= ap_start) & (dt_series <= ap_end)
        signals[~mask] = 0.0

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

    # Apply config overrides if provided
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    # Create adapter and run
    strategy_name = getattr(strategy, "name", type(strategy).__name__)
    adapter = _SignalAdapter(signals, warmup_bars=strategy.warmup, symbol=symbol, strategy_name=strategy_name, freq=freq)
    engine = EventDrivenBacktester(spec_manager=_get_specs(), config=config)
    result = engine.run(adapter, {symbol: bars})

    # ── Inject indicator panels into result metadata ─────────────────────
    _inject_indicator_panels(result, strategy, signals, datetimes)

    # ── Inject active_periods so report Buy-and-Hold matches OOS scope ──
    if active_periods is not None and len(active_periods) > 0:
        result.metadata["active_periods"] = active_periods

    # ── Inject direction so report can show 卖出持有 for short strategies ──
    if direction is not None:
        result.metadata["direction"] = direction

    return result


def run_on_regime_periods(
    strategy_class: type,
    params: dict[str, Any],
    symbol: str,
    regime_labels: list,
    split: str = "train",
    direction: str = "up",
    freq: str = "daily",
    signal_direction: str | None = None,
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
        signal_direction: Directional filter for signals: "long", "short", or None.

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
                direction=signal_direction,
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
