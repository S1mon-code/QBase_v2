#!/usr/bin/env python3
"""Iron Ore Long Portfolio — daily_v3 + 1h_v18 Signal Blending.

Fixed portfolio composition. Each strategy runs at its native frequency,
forecasts are resampled to 1H grid, blended with FDM, and executed as
one net position.

Daily component: continuous rebalancing (position updates when daily signal changes)
1H component: fixed entry (hold until signal flips)
Total position = daily_lots + hourly_lots
"""

from __future__ import annotations

import sys
import importlib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from pipeline.qbase_config import ALPHAFORGE_PATH as _AF_PATH, PROJECT_ROOT

for p in (str(_PROJECT_ROOT), _AF_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.engine.event_driven import EventDrivenBacktester
from alphaforge.engine.config import BacktestConfig
from alphaforge.report import HTMLReportGenerator
from pipeline.backtest_runner import _SignalAdapter
from portfolio.signal_blender import (
    compute_forecast_scalar, scale_and_cap,
    FORECAST_CAP,
)

# ── Portfolio Configuration ───────────────────────────────────────────────────

INSTRUMENT = 'I'
INITIAL_CAPITAL = 10_000_000
TARGET_VOL = 0.15
DAILY_RISK_SHARE = 0.55   # 55% of risk budget → daily
HOURLY_RISK_SHARE = 0.45  # 45% of risk budget → 1h
MAX_MARGIN_UTIL = 0.80

PORTFOLIO = [
    {
        'key': 'daily_v27',
        'freq': 'daily',
        'module': 'strategies.strong_trend.long.I.daily.trend_medium_v27',
        'risk_share': DAILY_RISK_SHARE,
        'sizing_mode': 'continuous',
    },
    {
        'key': '1h_v18',
        'freq': '1h',
        'module': 'strategies.strong_trend.long.I.1h.v18',
        'risk_share': HOURLY_RISK_SHARE,
        'sizing_mode': 'fixed_entry',
    },
]


def load_periods():
    labels = yaml.safe_load((_PROJECT_ROOT / 'data' / 'regime_labels' / 'I.yaml').read_text())['labels']
    oos_up = [l for l in labels if l['split'] == 'oos' and l['direction'] == 'up']
    train_up = [l for l in labels if l['split'] == 'train' and l['direction'] == 'up']
    return {
        'oos_active': [{"start": l['start'], "end": l['end']} for l in oos_up],
        'oos_start': min(l.get('buffer_start', l['start']) for l in oos_up),
        'oos_end': max(l.get('buffer_end', l['end']) for l in oos_up),
        'train_active': [{"start": l['start'], "end": l['end']} for l in train_up],
        'train_start': min(l.get('buffer_start', l['start']) for l in train_up),
        'train_end': max(l.get('buffer_end', l['end']) for l in train_up),
    }


def generate_signals(cls, params, freq, start, end, active_periods, loader):
    """Generate raw signals + datetimes at native frequency with regime mask."""
    bars = loader.load(INSTRUMENT, freq=freq, start=start, end=end)
    closes = bars._close
    highs = bars._high
    lows = bars._low
    opens = bars._open
    volumes = bars._volume
    datetimes = bars._datetime
    oi = getattr(bars, '_oi', None)
    if oi is None or len(oi) != len(closes):
        oi = np.zeros(len(closes), dtype=np.float64)

    strategy = cls()
    for k, v in params.items():
        setattr(strategy, k, v)
    strategy.on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
    signals = strategy.generate_signals()

    # Active periods mask
    if active_periods:
        dt_series = pd.to_datetime(datetimes)
        mask = np.zeros(len(signals), dtype=bool)
        for ap in active_periods:
            mask |= (dt_series >= pd.Timestamp(ap["start"])) & (dt_series <= pd.Timestamp(ap["end"]))
        signals[~mask] = 0.0

    return signals, datetimes


def resample_to_1h(scaled, raw_dt, common_dt, freq):
    """Resample scaled forecasts to 1H grid. Daily → forward-fill."""
    n = len(common_dt)
    result = np.full(n, np.nan)

    if freq == '1h':
        raw_dt_pd = pd.to_datetime(raw_dt)
        for i, dt in enumerate(common_dt):
            match = np.where(raw_dt_pd == dt)[0]
            if len(match) > 0:
                result[i] = scaled[match[0]]
    else:
        # Daily → forward-fill across 1H bars
        raw_dt_pd = pd.to_datetime(raw_dt)
        daily_series = pd.Series(scaled, index=raw_dt_pd)
        for i, dt in enumerate(common_dt):
            mask = daily_series.index <= dt
            if mask.any():
                result[i] = float(daily_series[mask].iloc[-1])

    return result


def vol_target_lots(forecast, capital, risk_share, price, multiplier, ann_vol, margin_rate):
    """Convert forecast → target lots for one component.

    lots = (forecast / 10) × (capital × risk_share × target_vol) / (price × multiplier × ann_vol)
    """
    if price <= 0 or ann_vol <= 1e-8:
        return 0
    risk_capital = capital * risk_share * TARGET_VOL
    ideal = (forecast / 10.0) * risk_capital / (price * multiplier * ann_vol)
    # Margin cap for this component
    margin_per_lot = price * multiplier * margin_rate
    if margin_per_lot > 0:
        max_lots = capital * risk_share * MAX_MARGIN_UTIL / margin_per_lot
        ideal = np.clip(ideal, -max_lots, max_lots)
    return int(ideal)


def estimate_vol(closes, idx, lookback=20):
    start = max(0, idx - lookback)
    window = closes[start:idx + 1]
    if len(window) < 5:
        return 0.0
    rets = np.diff(window) / window[:-1]
    daily_vol = float(np.std(rets))
    return daily_vol * np.sqrt(252) if daily_vol > 1e-8 else 0.0


def main():
    print("=" * 60)
    print("  Iron Ore Long Portfolio: daily_v3 + 1h_v18")
    print("=" * 60)

    periods = load_periods()
    loader = MarketDataLoader(f"{_AF_PATH}/data/")
    specs = ContractSpecManager()
    spec = specs.get(INSTRUMENT)

    # Load 1H bars as execution grid
    bars_1h = loader.load(INSTRUMENT, freq='1h', start=periods['oos_start'], end=periods['oos_end'])
    common_dt = pd.to_datetime(bars_1h._datetime)
    n_bars = len(common_dt)
    closes_1h = np.asarray(bars_1h._close, dtype=np.float64)

    print(f"  1H grid: {n_bars} bars")

    # ── Generate forecasts per strategy ───────────────────────────────────

    component_lots = []
    per_strategy_forecasts = {}   # key → scaled forecast on 1H grid
    per_strategy_scalars = {}     # key → scalar

    for strat in PORTFOLIO:
        key = strat['key']
        freq = strat['freq']
        risk_share = strat['risk_share']
        sizing_mode = strat['sizing_mode']

        print(f"\n  {key} ({freq}, risk={risk_share:.0%}, mode={sizing_mode})")

        # Import strategy
        mod = importlib.import_module(strat['module'])
        cls = None
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (isinstance(obj, type) and hasattr(obj, 'regime')
                and hasattr(obj, '_generate_signal')
                and obj.__name__ not in ('QBaseStrategy', 'TrendingStrategy', 'MeanReversionStrategy')):
                cls = obj
                break

        # Load optimized params
        research_base = _PROJECT_ROOT / 'research' / 'strong_trend' / 'long' / 'I' / freq
        params = {}
        for vdir in research_base.iterdir():
            if vdir.is_dir() and vdir.name.startswith(key.split('_')[-1] + '_'):
                pf = vdir / 'params.yaml'
                if pf.exists():
                    params = yaml.safe_load(pf.read_text()).get('best_params', {})
                break

        # Training scalar
        train_sig, _ = generate_signals(
            cls, params, freq, periods['train_start'], periods['train_end'],
            periods['train_active'], loader,
        )
        scalar = compute_forecast_scalar(train_sig)
        print(f"    scalar={scalar:.1f}")

        # OOS signals → scale → resample to 1H
        oos_sig, oos_dt = generate_signals(
            cls, params, freq, periods['oos_start'], periods['oos_end'],
            periods['oos_active'], loader,
        )
        scaled = np.array([scale_and_cap(s, scalar) for s in oos_sig])
        forecast_1h = resample_to_1h(scaled, oos_dt, common_dt, freq)

        # Direction filter (long only)
        forecast_1h = np.maximum(0, np.nan_to_num(forecast_1h, nan=0.0))

        # Store for report metadata
        per_strategy_forecasts[key] = forecast_1h.tolist()
        per_strategy_scalars[key] = scalar

        # Convert forecast → target lots
        lots = np.zeros(n_bars, dtype=np.float64)

        if sizing_mode == 'continuous':
            # Compute target lots every bar, naturally stable for daily
            # (forecast_1h is forward-filled from daily, so only changes once/day)
            for i in range(20, n_bars):
                ann_vol = estimate_vol(closes_1h, i)
                lots[i] = vol_target_lots(
                    forecast_1h[i], INITIAL_CAPITAL, risk_share,
                    closes_1h[i], spec.multiplier, ann_vol, spec.margin_rate,
                )

        elif sizing_mode == 'fixed_entry':
            # Size on entry, hold until signal flips to zero
            current = 0
            threshold = 1.0  # forecast > 1.0 to enter (out of 20)
            for i in range(20, n_bars):
                f = forecast_1h[i]
                if f > threshold and current == 0:
                    # Entry
                    ann_vol = estimate_vol(closes_1h, i)
                    current = vol_target_lots(
                        f, INITIAL_CAPITAL, risk_share,
                        closes_1h[i], spec.multiplier, ann_vol, spec.margin_rate,
                    )
                elif f <= threshold and current != 0:
                    # Exit
                    current = 0
                lots[i] = current

        component_lots.append(lots)
        max_l = int(np.max(np.abs(lots)))
        avg_l = np.mean(np.abs(lots[lots != 0])) if np.any(lots != 0) else 0
        changes = np.sum(np.diff(lots) != 0)
        print(f"    max_lots={max_l}, avg_lots={avg_l:.0f}, position_changes={changes}")

    # ── Combine components ────────────────────────────────────────────────

    total_lots = component_lots[0] + component_lots[1]

    # Total margin cap
    for i in range(len(total_lots)):
        if closes_1h[i] <= 0:
            continue
        margin_per_lot = closes_1h[i] * spec.multiplier * spec.margin_rate
        if margin_per_lot > 0:
            max_lots = INITIAL_CAPITAL * MAX_MARGIN_UTIL / margin_per_lot
            if abs(total_lots[i]) > max_lots:
                total_lots[i] = int(np.sign(total_lots[i]) * max_lots)

    # ── Portfolio-level vol cap (1.5x target vol) ─────────────────────────
    from portfolio.signal_blender import portfolio_vol_cap, monitor_correlation_regime

    total_lots = portfolio_vol_cap(
        total_lots, closes_1h, spec.multiplier, INITIAL_CAPITAL,
        TARGET_VOL, max_vol_ratio=1.5,
    )
    print(f"\n  After vol cap: max={int(np.max(total_lots))}")

    # ── Correlation regime monitoring ─────────────────────────────────────
    if len(component_lots) >= 2:
        # Build returns matrix from component lot changes
        comp_returns = np.column_stack([
            np.diff(np.append(0, lots)) for lots in component_lots
        ])
        corr_regime = monitor_correlation_regime(
            comp_returns,
            [strat['key'] for strat in PORTFOLIO],
        )
        print(f"  Correlation regime: {corr_regime['regime']} (avg_corr={corr_regime['avg_correlation']:.3f})")
        if corr_regime['position_scale'] < 1.0:
            print(f"  ⚠ Scaling positions by {corr_regime['position_scale']:.0%} due to {corr_regime['regime']} correlation")
            total_lots = (total_lots * corr_regime['position_scale']).astype(int)

    print(f"  Combined: max={int(np.max(total_lots))}, changes={np.sum(np.diff(total_lots) != 0)}")

    # ── Run through AlphaForge ────────────────────────────────────────────

    # Convert total_lots to signal for _SignalAdapter
    # The adapter does its own vol-target sizing from signal,
    # so we normalize lots to signal [0, 1] where 1 = max observed lots
    max_pos = max(np.max(np.abs(total_lots)), 1)
    adapter_signals = total_lots / max_pos

    print(f"  Running AlphaForge backtest...")

    config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        volume_adaptive_spread=True,
        dynamic_margin=True,
        time_varying_spread=True,
        rollover_window_bars=20,
        margin_check_mode="daily",
        asymmetric_impact=True,
        detect_locked_limit=True,
    )

    # Use 1H fixed entry mode for the combined adapter
    # (we already computed the exact lots, adapter just needs to follow)
    adapter = _SignalAdapter(
        adapter_signals, warmup_bars=0, symbol=INSTRUMENT,
        strategy_name='I_LONG_PORTFOLIO', freq='1h',
    )
    engine = EventDrivenBacktester(spec_manager=specs, config=config)
    result = engine.run(adapter, {INSTRUMENT: bars_1h})

    if hasattr(result, 'metadata') and 'indicator_panels' in result.metadata:
        del result.metadata['indicator_panels']

    print(f"\n  {'=' * 50}")
    print(f"  Sharpe:  {result.sharpe:.3f}")
    print(f"  Return:  {result.total_return:+.2%}")
    print(f"  MaxDD:   {result.max_drawdown:.2%}")
    print(f"  Trades:  {result.n_trades}")
    print(f"  Calmar:  {result.calmar:.3f}")
    print(f"  {'=' * 50}")

    # ── Run individual strategy backtests for comparison ──────────────────

    print("\n  Running individual strategy backtests for comparison...")
    per_strategy_metrics = {}
    per_strategy_equity = {}
    strategy_links = {}

    report_dir = _PROJECT_ROOT / 'reports' / 'strong_trend' / 'long' / 'I'
    report_dir.mkdir(parents=True, exist_ok=True)
    for f in report_dir.glob('*.html'):
        f.unlink()

    reporter = HTMLReportGenerator()

    for strat in PORTFOLIO:
        key = strat['key']
        freq = strat['freq']
        mod = importlib.import_module(strat['module'])
        cls = None
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (isinstance(obj, type) and hasattr(obj, 'regime')
                and hasattr(obj, '_generate_signal')
                and obj.__name__ not in ('QBaseStrategy', 'TrendingStrategy', 'MeanReversionStrategy')):
                cls = obj
                break

        # Load params
        research_base = _PROJECT_ROOT / 'research' / 'strong_trend' / 'long' / 'I' / freq
        params = {}
        for vdir in research_base.iterdir():
            if vdir.is_dir() and vdir.name.startswith(key.split('_')[-1] + '_'):
                pf = vdir / 'params.yaml'
                if pf.exists():
                    params = yaml.safe_load(pf.read_text()).get('best_params', {})
                break

        try:
            from pipeline.backtest_runner import run_qbase_backtest
            r = run_qbase_backtest(
                cls, params, INSTRUMENT, freq,
                periods['oos_start'], periods['oos_end'],
                direction='long', active_periods=periods['oos_active'],
            )
            if hasattr(r, 'metadata') and 'indicator_panels' in r.metadata:
                del r.metadata['indicator_panels']

            per_strategy_metrics[key] = {
                'sharpe': round(r.sharpe, 3),
                'total_return': round(r.total_return, 4),
                'max_drawdown': round(r.max_drawdown, 4),
                'n_trades': r.n_trades,
                'calmar': round(r.calmar, 3),
            }

            # Normalized equity curve
            eq = r.equity_curve
            if hasattr(eq, 'values'):
                eq = eq.values
            eq = np.asarray(eq, dtype=np.float64)
            if len(eq) > 0 and eq[0] > 0:
                per_strategy_equity[key] = (eq / eq[0]).tolist()

            # Generate individual strategy report
            strat_bars = {INSTRUMENT: loader.load(INSTRUMENT, freq=freq,
                          start=periods['oos_start'], end=periods['oos_end'])}
            strat_report = str(report_dir / f'{key}.html')
            reporter.generate(r, strat_report, bar_data=strat_bars, freq=freq)
            strategy_links[key] = f'{key}.html'

            print(f"    {key}: Sharpe={r.sharpe:.3f} Return={r.total_return:+.2%}")
        except Exception as e:
            print(f"    {key}: ERROR {e}")

    # ── Build signal_blend metadata ───────────────────────────────────────

    # Blended forecast (sum of weighted forecasts, already computed)
    # Reconstruct from per_strategy_forecasts
    n_strats = len(PORTFOLIO)
    weight_per = 1.0 / n_strats
    blended_forecast = np.zeros(n_bars)
    for key in per_strategy_forecasts:
        blended_forecast += np.array(per_strategy_forecasts[key]) * weight_per

    # Forecast correlation matrix
    fc_matrix = np.column_stack([np.array(per_strategy_forecasts[s['key']]) for s in PORTFOLIO])
    corr = np.corrcoef(fc_matrix.T)
    corr = np.nan_to_num(corr, nan=0.0)
    forecast_corr = {}
    keys_list = [s['key'] for s in PORTFOLIO]
    for i, k1 in enumerate(keys_list):
        for j, k2 in enumerate(keys_list):
            forecast_corr[f"{k1}_vs_{k2}"] = round(float(corr[i, j]), 3)

    result.metadata['signal_blend'] = {
        'weights': {s['key']: round(weight_per, 4) for s in PORTFOLIO},
        'fdm': 1.0,  # not used in additive mode but required by spec
        'per_strategy_signals': per_strategy_forecasts,
        'blended_signal': blended_forecast.tolist(),
        'net_position': [int(x) for x in total_lots],
        'per_strategy_metrics': per_strategy_metrics,
        'per_strategy_equity': per_strategy_equity,
        'forecast_correlation': forecast_corr,
        'datetimes': common_dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
    }

    # ── Generate Signal Blending Report ───────────────────────────────────

    report_path = str(report_dir / 'portfolio.html')
    reporter.generate_signal_blend_report(
        result, report_path,
        bar_data={INSTRUMENT: bars_1h}, freq='1h',
        strategy_links=strategy_links,
    )
    print(f"\n  Report: {report_path}")

    # ── Save config ───────────────────────────────────────────────────────

    config_dir = _PROJECT_ROOT / 'portfolio' / 'strong_trend' / 'long' / 'I'
    config_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        'instrument': INSTRUMENT,
        'direction': 'long',
        'regime': 'strong_trend',
        'initial_capital': INITIAL_CAPITAL,
        'target_vol': TARGET_VOL,
        'result': {
            'sharpe': round(result.sharpe, 3),
            'total_return': round(result.total_return, 4),
            'max_drawdown': round(result.max_drawdown, 4),
            'n_trades': result.n_trades,
        },
        'components': [
            {
                'key': s['key'],
                'freq': s['freq'],
                'risk_share': s['risk_share'],
                'sizing_mode': s['sizing_mode'],
            }
            for s in PORTFOLIO
        ],
    }
    yaml.dump(config_data, open(config_dir / 'config.yaml', 'w'),
              default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  Config: {config_dir / 'config.yaml'}")
    print(f"\n  open {report_path}")


if __name__ == '__main__':
    main()
