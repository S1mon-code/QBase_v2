"""Portfolio Runner — Multi-frequency additive position blending.

Architecture:
    daily_component = vol_target(daily_blend_forecast, capital × daily_risk_budget)
    hourly_component = vol_target(hourly_blend_forecast, capital × hourly_risk_budget)
    total_position = daily_component + hourly_component

Each frequency group:
    - Blends its strategies via Carver forecast combination (scale → cap → FDM)
    - Sizes independently using its own risk budget
    - Daily: continuous rebalancing (position updates once per day)
    - 1H: fixed entry (position holds until signal flips)

The runner produces a single adapter_signals array at 1H frequency,
which AlphaForge executes as one net position.
"""

from __future__ import annotations

import sys
import importlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_AF_PATH = "/Users/simon/Desktop/AlphaForge"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

for _p in (str(_PROJECT_ROOT), _AF_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.engine.event_driven import EventDrivenBacktester
from alphaforge.engine.config import BacktestConfig
from pipeline.backtest_runner import _SignalAdapter
from portfolio.signal_blender import (
    compute_forecast_scalar, scale_and_cap, compute_fdm,
    estimate_forecast_correlation, blend_forecasts,
    apply_direction_filter, forecast_to_position,
    FORECAST_CAP, FORECAST_TARGET_ABS,
)


@dataclass
class ComponentConfig:
    """Configuration for one frequency component."""

    name: str  # e.g. "daily_blend"
    freq: str  # "daily" or "1h"
    risk_budget: float  # fraction of target_vol (e.g. 0.09 = 9%)
    sizing_mode: str  # "continuous" or "fixed_entry"
    strategy_keys: list[str] = field(default_factory=list)
    strategy_classes: list[type] = field(default_factory=list)
    strategy_params: list[dict] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioConfig:
    """Full portfolio configuration."""

    instrument: str
    direction: str  # "long" or "short"
    regime: str
    initial_capital: float = 10_000_000
    target_vol: float = 0.15
    max_margin_util: float = 0.80
    components: list[ComponentConfig] = field(default_factory=list)


class PortfolioRunner:
    """Run multi-frequency additive signal blending backtest."""

    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config
        self._loader = MarketDataLoader(f"{_AF_PATH}/data/")
        self._specs = ContractSpecManager()

    def run(
        self,
        oos_start: str,
        oos_end: str,
        oos_active: list[dict[str, str]],
        train_start: str,
        train_end: str,
        train_active: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Execute the full portfolio backtest.

        Returns dict with:
            result: BacktestResult
            bars: BarArray (1H)
            components_data: list of per-component metadata
            weights_data: dict for weights.yaml
        """
        cfg = self.config
        spec = self._specs.get(cfg.instrument)

        # Load 1H bars as the common execution grid
        bars_1h = self._loader.load(cfg.instrument, freq='1h', start=oos_start, end=oos_end)
        common_dt = pd.to_datetime(bars_1h._datetime)
        n_bars = len(common_dt)
        closes = np.asarray(bars_1h._close, dtype=np.float64)

        print(f"\n  1H grid: {n_bars} bars ({oos_start} → {oos_end})")

        # ── Per-component: generate forecasts + compute target lots ────────

        components_data = []
        component_lots = []  # each is np.ndarray of shape (n_bars,)

        for comp in cfg.components:
            print(f"\n  Component: {comp.name} ({comp.freq}, risk={comp.risk_budget:.0%}, mode={comp.sizing_mode})")

            # Generate forecasts for each strategy in this component
            n_strats = len(comp.strategy_keys)
            forecast_matrix = np.full((n_bars, n_strats), np.nan)
            scalars = {}

            for col, (key, cls, params) in enumerate(
                zip(comp.strategy_keys, comp.strategy_classes, comp.strategy_params)
            ):
                # Scalar from training data
                try:
                    train_sig = self._generate_signals(
                        cls, params, comp.freq, train_start, train_end, train_active,
                    )
                    scalar = compute_forecast_scalar(train_sig)
                except Exception:
                    scalar = 1.0
                scalars[key] = scalar

                # OOS signals at native frequency → resample to 1H
                try:
                    raw_sig, raw_dt = self._generate_signals_with_dt(
                        cls, params, comp.freq, oos_start, oos_end, oos_active,
                    )
                    scaled = np.array([scale_and_cap(s, scalar) for s in raw_sig])
                    forecast_matrix[:, col] = self._resample_to_1h(
                        scaled, raw_dt, common_dt, comp.freq,
                    )
                    valid = np.sum(~np.isnan(forecast_matrix[:, col]))
                    print(f"    {key}: scalar={scalar:.1f}, valid={valid}/{n_bars}")
                except Exception as e:
                    print(f"    {key}: ERROR {e}")

            # Compute FDM
            corr = estimate_forecast_correlation(forecast_matrix)
            fdm = compute_fdm(comp.weights, corr, comp.strategy_keys)
            print(f"    FDM={fdm:.3f}")

            # Blend forecasts bar-by-bar
            blended = np.zeros(n_bars)
            for i in range(n_bars):
                fcs = {comp.strategy_keys[j]: float(forecast_matrix[i, j]) for j in range(n_strats)}
                result = blend_forecasts(fcs, comp.weights, fdm=fdm, total_strategies=n_strats)
                blended[i] = apply_direction_filter(result.forecast, cfg.direction)

            # Convert forecast → target lots using component's risk budget
            comp_lots = np.zeros(n_bars, dtype=np.float64)
            vol_lookback = 20

            if comp.sizing_mode == "continuous":
                # Daily continuous: update target every bar, but only changes on new daily bar
                for i in range(vol_lookback, n_bars):
                    ann_vol = self._estimate_vol(closes, i, vol_lookback)
                    if ann_vol > 0:
                        comp_lots[i] = forecast_to_position(
                            blended[i], cfg.initial_capital, comp.risk_budget,
                            closes[i], spec.multiplier, ann_vol,
                            margin_rate=spec.margin_rate,
                        )

            elif comp.sizing_mode == "fixed_entry":
                # Fixed entry: size on entry, hold until signal flips
                current_lots = 0
                signal_threshold = 0.05 * FORECAST_CAP  # 1.0 in forecast scale
                for i in range(vol_lookback, n_bars):
                    forecast = blended[i]
                    if abs(forecast) > signal_threshold and current_lots == 0:
                        # Entry
                        ann_vol = self._estimate_vol(closes, i, vol_lookback)
                        if ann_vol > 0:
                            current_lots = forecast_to_position(
                                forecast, cfg.initial_capital, comp.risk_budget,
                                closes[i], spec.multiplier, ann_vol,
                                margin_rate=spec.margin_rate,
                            )
                    elif abs(forecast) <= signal_threshold and current_lots != 0:
                        # Exit
                        current_lots = 0
                    elif current_lots != 0 and (
                        (current_lots > 0 and forecast < -signal_threshold)
                        or (current_lots < 0 and forecast > signal_threshold)
                    ):
                        # Direction flip → re-entry
                        ann_vol = self._estimate_vol(closes, i, vol_lookback)
                        if ann_vol > 0:
                            current_lots = forecast_to_position(
                                forecast, cfg.initial_capital, comp.risk_budget,
                                closes[i], spec.multiplier, ann_vol,
                                margin_rate=spec.margin_rate,
                            )
                    comp_lots[i] = current_lots

            component_lots.append(comp_lots)
            components_data.append({
                "name": comp.name,
                "freq": comp.freq,
                "risk_budget": comp.risk_budget,
                "sizing_mode": comp.sizing_mode,
                "fdm": fdm,
                "scalars": scalars,
                "strategies": comp.strategy_keys,
                "position_series": list(zip(common_dt.tolist(), comp_lots.tolist())),
                "forecast_series": list(zip(common_dt.tolist(), blended.tolist())),
            })
            print(f"    Max lots: {int(np.max(np.abs(comp_lots)))}")

        # ── Combine components ─────────────────────────────────────────────

        total_lots = sum(component_lots)

        # Apply total margin cap
        for i in range(len(total_lots)):
            if closes[i] <= 0:
                continue
            margin_per_lot = closes[i] * spec.multiplier * spec.margin_rate
            if margin_per_lot > 0:
                max_lots = cfg.initial_capital * cfg.max_margin_util / margin_per_lot
                if abs(total_lots[i]) > max_lots:
                    scale = max_lots / abs(total_lots[i])
                    total_lots[i] = int(total_lots[i] * scale)

        # Convert total_lots to signal [0, 1] for adapter
        # Normalize by the max observed position
        max_pos = max(np.max(np.abs(total_lots)), 1)
        adapter_signals = total_lots / max_pos

        print(f"\n  Total: max_lots={int(np.max(np.abs(total_lots)))}, "
              f"avg_lots={np.mean(np.abs(total_lots[total_lots != 0])):.0f}")

        # ── Run AlphaForge backtest ────────────────────────────────────────

        print("  Running AlphaForge backtest...")
        config = BacktestConfig(
            initial_capital=cfg.initial_capital,
            volume_adaptive_spread=True,
            dynamic_margin=True,
            time_varying_spread=True,
            rollover_window_bars=20,
            margin_check_mode="daily",
            asymmetric_impact=True,
            detect_locked_limit=True,
        )

        # Portfolio uses continuous rebalancing at 1H
        # but with 10% buffer to avoid excessive trading
        adapter = _SignalAdapter(
            adapter_signals, warmup_bars=0, symbol=cfg.instrument,
            strategy_name='PORTFOLIO', freq='daily',  # continuous mode
        )
        engine = EventDrivenBacktester(spec_manager=self._specs, config=config)
        result = engine.run(adapter, {cfg.instrument: bars_1h})

        # Inject component metadata
        result.metadata['portfolio_components'] = {
            "components": [
                {k: v for k, v in cd.items() if k != 'position_series' and k != 'forecast_series'}
                for cd in components_data
            ],
        }

        # Clear indicator_panels if present
        if 'indicator_panels' in result.metadata:
            del result.metadata['indicator_panels']

        print(f"\n  Result: Sharpe={result.sharpe:.3f} Return={result.total_return:+.2%} "
              f"MaxDD={result.max_drawdown:.2%} Trades={result.n_trades}")

        return {
            "result": result,
            "bars": bars_1h,
            "components_data": components_data,
            "total_lots": total_lots,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _generate_signals(
        self, cls, params, freq, start, end, active_periods,
    ) -> np.ndarray:
        """Generate raw signal array at native frequency."""
        bars = self._loader.load(self.config.instrument, freq=freq, start=start, end=end)
        strategy = cls()
        for k, v in params.items():
            setattr(strategy, k, v)

        closes = bars._close
        highs = bars._high
        lows = bars._low
        opens = bars._open
        volumes = bars._volume
        datetimes = bars._datetime
        oi = getattr(bars, '_oi', None)
        if oi is None or len(oi) != len(closes):
            oi = np.zeros(len(closes), dtype=np.float64)

        strategy.on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        signals = strategy.generate_signals()

        # Apply active_periods mask
        if active_periods:
            dt_series = pd.to_datetime(datetimes)
            mask = np.zeros(len(signals), dtype=bool)
            for ap in active_periods:
                mask |= (dt_series >= pd.Timestamp(ap["start"])) & (dt_series <= pd.Timestamp(ap["end"]))
            signals[~mask] = 0.0

        return signals

    def _generate_signals_with_dt(
        self, cls, params, freq, start, end, active_periods,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate raw signals + datetime array."""
        bars = self._loader.load(self.config.instrument, freq=freq, start=start, end=end)
        strategy = cls()
        for k, v in params.items():
            setattr(strategy, k, v)

        closes = bars._close
        highs = bars._high
        lows = bars._low
        opens = bars._open
        volumes = bars._volume
        datetimes = bars._datetime
        oi = getattr(bars, '_oi', None)
        if oi is None or len(oi) != len(closes):
            oi = np.zeros(len(closes), dtype=np.float64)

        strategy.on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        signals = strategy.generate_signals()

        if active_periods:
            dt_series = pd.to_datetime(datetimes)
            mask = np.zeros(len(signals), dtype=bool)
            for ap in active_periods:
                mask |= (dt_series >= pd.Timestamp(ap["start"])) & (dt_series <= pd.Timestamp(ap["end"]))
            signals[~mask] = 0.0

        return signals, datetimes

    @staticmethod
    def _resample_to_1h(
        scaled: np.ndarray,
        raw_dt: np.ndarray,
        common_dt: pd.DatetimeIndex,
        freq: str,
    ) -> np.ndarray:
        """Resample signals to 1H grid. Daily signals are forward-filled."""
        n = len(common_dt)
        result = np.full(n, np.nan)

        if freq == '1h':
            raw_dt_pd = pd.to_datetime(raw_dt)
            for i, dt in enumerate(common_dt):
                match = np.where(raw_dt_pd == dt)[0]
                if len(match) > 0:
                    result[i] = scaled[match[0]]
        else:
            # Daily → forward-fill
            raw_dt_pd = pd.to_datetime(raw_dt)
            daily_series = pd.Series(scaled, index=raw_dt_pd)
            for i, dt in enumerate(common_dt):
                mask = daily_series.index <= dt
                if mask.any():
                    result[i] = float(daily_series[mask].iloc[-1])

        return result

    @staticmethod
    def _estimate_vol(closes: np.ndarray, idx: int, lookback: int) -> float:
        """Estimate annualized volatility from recent closes."""
        start = max(0, idx - lookback)
        window = closes[start:idx + 1]
        if len(window) < 5:
            return 0.0
        rets = np.diff(window) / window[:-1]
        daily_vol = float(np.std(rets))
        return daily_vol * np.sqrt(252) if daily_vol > 1e-8 else 0.0
