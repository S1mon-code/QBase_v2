# QBase_v2 — Agent 开发指南

黑色系中国期货单品种多策略系统。基本面方向约束 + 技术面 regime 适配 + 多周期策略库。

**复用：** indicators/（324 + 新增 Carry）+ AlphaForge V7.1。其余全部重写。

---

## 架构总览

```
基本面团队预判 Regime + 方向（周/月级更新）
         │
    ┌────▼────┐
    │ Layer 0 │  Fundamental View — direction + regime per instrument
    └────┬────┘
    ┌────▼────┐
    │ Layer 1 │  Historical Regime Matching — 匹配历史同类时段(±2月buffer)
    └────┬────┘
    ┌────▼────┐
    │ Layer 2 │  Strategy Pool — 按 Regime 激活对应策略集
    │         │  Trending: TSMOM Baselines + Momentum + Carry + Blended + Technical
    │         │  Mean Reversion: BB / RSI / Keltner / CCI / Carry MR
    │         │  Crisis: 减仓 + 收紧止损
    └────┬────┘
    ┌────▼────┐
    │ Layer 3 │  Signal Blender — 多策略信号加权合并 → 单一净信号
    └────┬────┘
    ┌────▼────┐
    │ Layer 4 │  Directional Filter — 看多→max(0,signal), 看空→min(0,signal)
    └────┬────┘
    ┌────▼────┐
    │ Layer 5 │  Vol Targeting + Position Sizing
    └────┬────┘
    ┌────▼────┐
    │ Layer 6 │  Chandelier Exit (regime-adaptive)
    └────┬────┘
    ┌────▼────┐
    │ Layer 7 │  Portfolio Stops — 预警-10% / 减仓-15% / 熔断-20% / 单日-5%
    └────┬────┘
    ┌────▼────┐
    │ Layer 8 │  Execution — AlphaForge V7.1 Industrial 模式
    └─────────┘
```

---

## 开发 Phase 总览

| Phase | 名称 | 依赖 | 详细文档 |
|:-----:|------|------|---------|
| 1 | 项目骨架 + 数据基础 | — | [phase01-skeleton.md](docs/phases/phase01-skeleton.md) |
| 2 | Regime 标注系统 + 数据分割 | Phase 1 | [phase02-regime-labeling.md](docs/phases/phase02-regime-labeling.md) |
| 3 | 风控模块 | Phase 1 | [phase03-risk-management.md](docs/phases/phase03-risk-management.md) |
| 4 | 策略模板 + 第一批策略（全周期） | Phase 2, 3 | [phase04-strategy-development.md](docs/phases/phase04-strategy-development.md) |
| 5 | 优化器 | Phase 4 | [phase05-optimizer.md](docs/phases/phase05-optimizer.md) |
| 6 | 验证体系 | Phase 5 | [phase06-validation.md](docs/phases/phase06-validation.md) |
| 7 | 归因分析 | Phase 6 | [phase07-attribution.md](docs/phases/phase07-attribution.md) |
| 8 | Portfolio 构建 | Phase 7 | [phase08-portfolio.md](docs/phases/phase08-portfolio.md) |
| 9 | 扩展品种 | Phase 8 | [phase09-expand-instruments.md](docs/phases/phase09-expand-instruments.md) |
| 10 | Pipeline + CLI + Reporting | Phase 4+ | [phase10-pipeline-cli.md](docs/phases/phase10-pipeline-cli.md) |
| 11 | 监控 + 实盘部署 | Phase 8 | [phase11-monitoring-deployment.md](docs/phases/phase11-monitoring-deployment.md) |

**关键路径：** Phase 1 → 2+3(并行) → 4 → 5 → 6 → 7 → 8

**Phase 10 可从 Phase 4 开始逐步搭建。**

**注意：** 策略开发（Phase 4）即在 1h/2h/4h/daily 全周期上同步进行，不单独设"扩展周期"Phase。每个策略从一开始就在多周期上验证。

---

## 核心参数

| 维度 | 规格 |
|------|------|
| 品种 | 黑色系：RB → HC → I → J → JM |
| 周期 | 1h / 2h / 4h / daily 同步开发 |
| Regime | Strong Trend / Mild Trend / Mean Reversion / Crisis |
| 信号维度 | Momentum / Carry / Volume-OI / Technical |
| Trend Horizon | Fast(20-60) / Medium(60-125) / Slow(125-250) |
| 方向约束 | 基本面 view: LONG_ONLY / SHORT_ONLY / NEUTRAL |
| 风控 | Chandelier Exit (regime-adaptive) + Vol Targeting + 2% 单笔风险 |
| 优化函数 | 5 维复合: Performance(40%) + Significance(15%) + Consistency(15%) + Risk(15%) + Alpha(15%) |
| 验证 | 6 层: Regime CV → OOS → Walk-Forward → Deflated Sharpe → Monte Carlo → Industrial |
| 归因 | 5 层: Signal → Horizon → Regime → Baseline Decomposition → Operational |
| Portfolio | Signal Blender(同Regime内) + Regime Allocation(跨Regime) |
| 标注方法 | Bry-Boschan 初筛 + 人工校正, ±2月buffer, 阈值 5%/20% |
| 切换频率 | 周/月级 |
| 数据分割 | 标注时即分好 train/oos/holdout per regime |

---

## 目标项目结构

```
QBase_v2/
├── CLAUDE.md                       # 本文件
├── config/
│   ├── settings.yaml               # AlphaForge 路径、全局参数
│   ├── fundamental_views.yaml      # 基本面方向约束
│   └── regime_thresholds.yaml      # Regime 标注阈值
├── data/
│   └── regime_labels/              # 历史 regime 标注 YAML (per instrument)
├── indicators/                     # 324 + 新增 Carry 指标
├── regime/                         # Regime 标注 + 匹配
├── strategies/
│   ├── templates/                  # 策略模板 (trending / mean_reversion)
│   ├── baselines/                  # TSMOM Baselines (fast/medium/slow)
│   ├── trending/
│   │   ├── fast/
│   │   ├── medium/
│   │   └── slow/
│   └── mean_reversion/
├── risk/                           # 风控模块
├── optimizer/                      # 优化器
├── validation/                     # 验证体系
├── attribution/                    # 归因分析
├── portfolio/                      # Portfolio 构建
├── pipeline/                       # 流水线编排 + CLI
├── monitoring/                     # 监控 + 实盘
├── reports/                        # HTML 报告
├── research_log/                   # 实验记录
│   └── trials/                     # 全部试验记录 (Deflated Sharpe 用)
├── tests/                          # 单元测试
└── docs/
    └── phases/                     # 各 Phase 详细设计文档
```

---

## 当前实现状态 (截至 2026-04-01)

| Phase | 状态 | 完成度 |
|-------|------|--------|
| 1 — 项目骨架 | ✅ 完成 | 100% |
| 2 — Regime 标注系统 | ✅ 完成 | 100%（I + AG long/short 已标注）|
| 3 — 风控模块 | ✅ 完成 | 100% |
| 4 — 策略开发 | ✅ 完成 | 100%（270 策略：I + AG，4 timeframe）|
| 5 — 优化器 | ✅ 完成 | 100% |
| 6 — 验证体系 | ✅ 完成 | 100% |
| 7 — 归因分析 | ✅ 完成 | 100% |
| 8 — Portfolio 构建 | ✅ 完成 | 100% |
| 9 — 扩展品种 | ✅ I + AG 完成 | 50%（RB/HC/J/JM 待开发）|
| 10 — Pipeline + CLI | ✅ 完成 | 95%（batch pipeline + 自动报告命名）|
| 11 — 监控 + 实盘 | 🔄 进行中 | 60%（缺 paper trading）|

**测试覆盖：** 576+ tests, 100% pass rate

### 策略规模

| Group | 品种 | 方向 | Regime | Timeframes | 策略数 | 通过验证 |
|-------|------|------|--------|-----------|--------|---------|
| strong_trend/long/I | I | long | strong_trend | daily/1h/2h/4h | 110 | 102 |
| strong_trend/long/AG | AG | long | strong_trend | daily/1h/2h/4h | 40 | 35 |
| strong_trend/short/AG | AG | short | strong_trend | daily/1h/2h/4h | 40 | 27 |
| mild_trend/long/I | I | long | mild_trend | daily/1h/2h/4h | 40 | 待标注 |
| mild_trend/short/I | I | short | mild_trend | daily/1h/2h/4h | 40 | 18 |

### Research 文件夹命名

研究结果目录格式：`research/{regime}/{direction}/{instrument}/{timeframe}/v{N}_{+/-}{return}%/`

- return 从 `oos.html` 的「总收益」字段提取，保留两位小数，正数带 `+`
- 示例：`research/strong_trend/long/AG/1h/v10_+97.98%/`
- OOS 包含该品种/方向下**所有** `split=oos` 的 regime periods（不按 regime 筛选）
- `run_single_strategy_pipeline()` 自动从 oos.html 提取总收益并命名

### 下一步优先级

1. mild_trend/long/I 的 regime 标注补全
2. RB/HC/J/JM 品种扩展
3. Paper Trading 验证
4. 完整 CLI 集成

---

## 全局规则

### 预计算模式（必须）

所有策略必须使用 `on_init_arrays` 预计算。`on_bar` 通过 `bar_index` 查表。

### 参数约束

- 每策略可优化参数 ≤ 5 个（含 chandelier_mult）
- 范围窄 (2-3x)
- 风控参数和信号参数一起优化

### 回测模式

- 开发/粗调: Basic 模式（快速迭代）
- 精调/验证: Industrial 模式（必须）
- Industrial 衰减 > 50% 的策略不入 Portfolio

### 测试集只读

不能因测试集结果修改参数。数据分割在 Phase 2 标注时完成，终身不变。

### 试验记录

优化器每次 trial 自动写入 `research_log/trials/`，不可删除。Deflated Sharpe 计算依赖完整记录。

### 策略命名

`{regime}_{horizon}_v{N}` — 如 `trend_medium_v1`, `mr_v3`

### Git Commit 规范

```
[模块] 类型: 简短描述
示例:
[regime] feat: auto labeler with Bry-Boschan
[strategy] feat: trend_medium_v1 SuperTrend+VolMom
[optimizer] fix: boundary protection for edge params
```

---

## AlphaForge V7.1 集成指南

AlphaForge 是 QBase_v2 的回测执行引擎（Layer 8）。本节提供完整 API 参考，供开发者在 QBase 中直接调用。

**AlphaForge 路径**：`/Users/simon/Desktop/AlphaForge`（需加入 `PYTHONPATH`）

### 版本特性概览

| 版本 | 核心新增 |
|------|---------|
| V7.1 | Purged K-Fold CV / WF+Optuna 联合优化 / ML Pipeline (LightGBM/XGBoost) |
| V7.0 | 止盈止损订单系统 / DecayDetector / ExperimentTracker / LiveGateway 骨架 |
| V6.0 | 工业级真实度：动态保证金 / 渐进换月 / 锁板检测 / 方向不对称冲击 |

---

### 快速接入

```python
import sys
sys.path.insert(0, "/Users/simon/Desktop/AlphaForge")

from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.engine.event_driven import EventDrivenBacktester
from alphaforge.engine.config import BacktestConfig

# 加载数据
loader = MarketDataLoader("/Users/simon/Desktop/AlphaForge/data/")
bars = loader.load("RB", freq="daily", start="2013-01-01", end="2023-12-31")

# 运行策略
config = BacktestConfig(initial_capital=10_000_000)
engine = EventDrivenBacktester(spec_manager=ContractSpecManager(), config=config)
result = engine.run(my_strategy, {"RB": bars})
result.print_summary()
```

---

### BacktestContext API

QBase 策略的 `on_bar(context)` / `_generate_signal` 查表时使用：

#### 价格属性（零 Bar 创建开销）

| 属性 | 类型 | 说明 |
|------|------|------|
| `context.close_raw` | float | 当前原始收盘价 |
| `context.open_raw` | float | 当前原始开盘价 |
| `context.high_raw` | float | 当前原始最高价 |
| `context.low_raw` | float | 当前原始最低价 |
| `context.close` | float | 复权收盘价 |
| `context.volume` | float | 当前成交量 |
| `context.settlement` | float | 结算价 |
| `context.is_rollover` | bool | 是否换月日 |
| `context.bar_index` | int | **预计算查表索引**（QBase 关键属性） |
| `context.datetime` | datetime64 | 当前 bar 时间 |

#### 持仓与权益

```python
side, lots = context.position   # side: 1=多, -1=空, 0=无; lots: 手数
context.equity                  # 当前权益（cash + 浮动盈亏）
context.available_cash          # 可用资金
```

#### 下单（信号在下一个 bar 的 open 执行）

```python
context.buy(lots)              # 开多
context.sell(lots)             # 开空
context.close_long()           # 平多（全部）
context.close_long(lots=5)     # 平多（指定手数）
context.close_short()          # 平空
```

#### 历史数据（最近 N 根 bar）

```python
context.get_close_array(n)     # np.ndarray, shape=(n,)
context.get_high_array(n)
context.get_low_array(n)
context.get_open_array(n)
context.get_volume_array(n)
```

#### 完整数据（用于 on_init_arrays 预计算）

```python
context.get_full_close_array()       # shape=(N,), 完整复权 close
context.get_full_high_array()
context.get_full_low_array()
context.get_full_open_array()
context.get_full_volume_array()
context.get_full_oi_array()
context.get_full_close_raw_array()
context.get_full_settlement_array()
context.get_full_datetime_array()
context.get_bars()                   # 当前品种完整 BarArray
```

---

### BacktestConfig 推荐配置

```python
from alphaforge.engine.config import BacktestConfig

# 开发/粗调（快速迭代）
config_dev = BacktestConfig(initial_capital=10_000_000)

# 精调/验证（工业级，QBase 必须）
config_industrial = BacktestConfig(
    initial_capital=10_000_000,
    volume_adaptive_spread=True,      # 按成交量动态调整价差
    dynamic_margin=True,              # 交割月阶梯保证金
    time_varying_spread=True,         # 按时段调整价差（开盘/收盘加宽）
    rollover_window_bars=20,          # Sigmoid 渐进换月（20 bar 窗口）
    margin_check_mode="daily",        # 每日结算价检查保证金
    margin_call_grace_bars=3,         # 追保宽限期 3 bar
    asymmetric_impact=True,           # 方向不对称冲击
    detect_locked_limit=True,         # 锁板检测
)

# Optuna 优化（抑制日志）
config_optuna = BacktestConfig(
    initial_capital=10_000_000,
    safe_mode=True,
    suppress_order_logs=True,
)
```

---

### BacktestResult 指标参考

```python
result = engine.run(strategy, bars)

# 核心指标
result.total_return          # 总收益率
result.annualized_return     # 年化收益率
result.sharpe                # 年化 Sharpe (rf=0)
result.sortino               # Sortino
result.calmar                # Calmar = 年化收益 / 最大回撤
result.max_drawdown          # 最大回撤（0.15 = 15%）
result.max_drawdown_duration # 最大回撤持续天数
result.volatility            # 年化波动率
result.win_rate              # 正收益日占比

# 扩展指标
result.cvar_95               # CVaR — 最差 5% 日收益均值
result.omega_ratio           # Omega 比率
result.tail_ratio            # 尾部比率
result.profit_factor         # 盈亏比
result.n_trades              # 交易次数

# 数据
result.equity_curve          # pd.Series 日权益曲线
result.daily_returns         # pd.Series 日收益率
result.trades                # pd.DataFrame 交易明细

# V7.0 衰减报告（>= 252 天数据时自动生成）
result.decay_report          # DecayReport 或 None
result.stability_score       # 0-100 稳定性评分，-1 = 数据不足
```

---

### 止盈止损订单系统（V7.0）

QBase Layer 6 Chandelier Exit 可接入此系统：

```python
# 基础止损止盈
context.set_stop_loss(price * 0.98)      # 绝对价格止损
context.set_take_profit(price * 1.05)    # 绝对价格止盈
context.set_trailing_stop(0.03)          # 从最高点回撤 3% 触发
context.cancel_stop_orders()             # 取消所有止损单

# Bracket 一键开仓+止损+止盈（百分比从成交价计算）
context.buy_bracket(lots, stop_loss_pct=0.02, take_profit_pct=0.05)
context.sell_bracket(lots, stop_loss_pct=0.02, take_profit_pct=0.05)
context.buy_bracket(lots, trailing_stop_pct=0.03)
```

**执行规则**：
- 止损触发检查在每个 bar 的 high/low 上进行（非 close）
- 触发后同 bar 提交平仓，下一 bar open 执行
- 手动 `close_long()` / `close_short()` 自动清除所有止损单

---

### 策略衰减检测（V7.0）

对应 QBase monitoring 模块：

```python
from alphaforge.analytics.decay import DecayDetector

detector = DecayDetector()
report = detector.analyze(result.daily_returns, result.equity_curve)

report.stability_score       # 0-100，< 50 = 不稳定
report.rolling_sharpe_slope  # 负值 = 衰减中
report.sharpe_half_life      # Alpha 半衰期（天）
report.current_regime        # 'low_vol' / 'normal' / 'high_vol' / 'crisis'
report.regime_sharpe         # 各 Regime 下的 Sharpe
report.break_dates           # 结构性断裂日期列表
report.warnings              # 人类可读预警
report.print_summary()
```

**QBase 规则**：`stability_score < 50` 或 Industrial 衰减 > 50% 的策略不入 Portfolio。

---

### 实验追踪（V7.0）

对应 QBase `research_log/trials/`：

```python
from alphaforge.store.tracker import ExperimentTracker

tracker = ExperimentTracker()     # ~/.alphaforge/experiments.db

with tracker.track("trend_medium_v3", tags=["trending", "medium"]) as run:
    result = engine.run(strategy, bars)
    run.log_result(result)
    run.log_params({"macd_fast": 12, "macd_slow": 26, "oi_period": 14})
    run.log_config(config)
    run.log_note("MACD line + OI Flow")

# 查询
tracker.best_run("trend_medium_v3", metric="sharpe")
tracker.compare_experiments("trend_medium_v1", "trend_medium_v2", "trend_medium_v3")
```

```bash
af experiment list
af experiment best trend_medium_v3 -m sharpe
af experiment compare trend_medium_v1 trend_medium_v2 trend_medium_v3
```

---

### Purged K-Fold 交叉验证（V7.1）

对应 QBase `validation/regime_cv`：

```python
from alphaforge.analytics.purged_kfold import PurgedKFoldAnalyzer

analyzer = PurgedKFoldAnalyzer(n_splits=5, purge_bars=20, embargo_bars=5)
report = analyzer.analyze(
    strategy_factory=lambda: TrendMediumV3(),
    bars={"RB": bars},
    spec_manager=ContractSpecManager(),
    initial_capital=1_000_000,
)

report.mean_oos_sharpe       # OOS 各折平均 Sharpe
report.overfit_probability   # 0-1，> 0.5 = 明显过拟合
report.oos_degradation       # IS/OOS 比值，越接近 1 越稳健
report.fold_metrics          # 各折详细 IS/OOS 指标
report.print_summary()
```

```bash
af purged-cv strategy.py --symbols RB --folds 5 --purge 20 --embargo 5
```

**purge / embargo**：purge = 清除 val 折前的泄漏窗口；embargo = 清除 val 折后的自相关窗口。

---

### WF + Optuna 联合优化（V7.1）

对应 QBase `optimizer/`：

```python
from alphaforge.optimizer.walk_forward_optimizer import WalkForwardOptimizer

opt = WalkForwardOptimizer(
    strategy_file="strategies/trending/medium/v3.py",
    symbols=["RB"],
    freq="daily",
    start="2013-01-01",
    end="2023-12-31",
    train_years=5,
    test_years=1,
    step_years=1,
)
opt.add_param("macd_fast", 8, 16, step=1)
opt.add_param("macd_slow", 20, 32, step=2)
opt.add_param("chandelier_mult", 2.0, 3.5, step=0.25)

result = opt.optimize(n_trials=50, objective="sharpe")

result.param_stability           # {"macd_fast": 0.12, ...} CV < 0.2 = 稳定
result.mean_oos_value            # 平均 OOS Sharpe
result.oos_degradation           # IS/OOS 比值
```

```bash
af wf-optimize strategy.py --symbols RB --train-years 5 --test-years 1 --trials 50
```

---

### ML Pipeline（V7.1）

```python
from alphaforge.ml import MLPipeline, FeatureEngineer, LGBMModel

fe = FeatureEngineer(
    factor_names=["returns_5d", "returns_20d", "volatility_20d", "volume_ratio"],
    holding_period=5,
    winsorize_sigma=3.0,
    normalize=True,
)
pipeline = MLPipeline(model=LGBMModel(), feature_engineer=fe, n_splits=5, purge_bars=20)
report = pipeline.fit_predict({"RB": bars_RB, "I": bars_I}, spec_manager=ContractSpecManager())

report.mean_ic                 # Spearman IC，> 0.03 = 有效因子
report.feature_importances     # {"returns_5d": 0.35, ...}
```

内置 7 个特征：`returns_1d`, `returns_5d`, `returns_20d`, `volatility_20d`, `volume_ratio`, `price_vs_sma20`, `atr_ratio`。

---

### 数据加载（黑色系）

```python
from alphaforge.data.market import MarketDataLoader

loader = MarketDataLoader("/Users/simon/Desktop/AlphaForge/data/")

# 黑色系单品种
bars_rb = loader.load("RB", freq="daily", start="2013-01-01")
bars_hc = loader.load("HC", freq="daily", start="2013-01-01")
bars_i  = loader.load("I",  freq="daily", start="2013-01-01")
bars_j  = loader.load("J",  freq="daily", start="2013-01-01")
bars_jm = loader.load("JM", freq="daily", start="2013-01-01")

# 多周期（1h/4h/daily）
bars_1h   = loader.load("RB", freq="1h")
bars_4h   = loader.load("RB", freq="4h")
bars_daily = loader.load("RB", freq="daily")

# 多品种 Panel（ML / 因子研究）
panel = loader.load_panel(["RB", "HC", "I", "J", "JM"], freq="daily", start="2015")
```

---

### 黑色系合约参数

| 品种 | 乘数 | Tick | 保证金率 | 价格限制 |
|------|------|------|---------|---------|
| RB（螺纹钢） | 10 | 1.0 | 0.10 | 0.05 |
| HC（热卷）  | 10 | 1.0 | 0.10 | 0.05 |
| I（铁矿石） | 100 | 0.5 | 0.12 | 0.08 |
| J（焦炭）   | 100 | 0.5 | 0.15 | 0.08 |
| JM（焦煤）  | 60  | 0.5 | 0.15 | 0.08 |

```python
from alphaforge.data.contract_specs import ContractSpecManager
specs = ContractSpecManager()
spec = specs.get("I")
specs.calc_commission("I", price=800, lots=10, is_open=True)
specs.calc_margin("I", price=800, lots=10)
```

---

### Numba 指标库

AlphaForge 内置 Numba 加速指标（`@njit(cache=True)`，首次 ~0.5s，后续 C 速度）：

```python
from alphaforge.indicators import sma, ema, rsi, atr, bollinger_bands, macd, supertrend, crossover, crossunder

sma(close, period)                              # → np.ndarray
ema(close, period)                              # → np.ndarray
rsi(close, period=14)                           # → np.ndarray
atr(high, low, close, period=14)                # → np.ndarray
bollinger_bands(close, period=20, num_std=2.0)  # → (upper, mid, lower)
macd(close, fast=12, slow=26, signal=9)         # → (line, signal, hist)
supertrend(high, low, close, period=10, mult=3) # → (values, direction)
crossover(a, b)                                 # → bool array (a 上穿 b)
crossunder(a, b)                                # → bool array (a 下穿 b)
```

**QBase 策略使用原则**：优先使用 `indicators/` 中已有的 QBase 指标（324+）；只在 QBase 无对应实现时才使用 AlphaForge 内置 Numba 指标。

---

### 完整 CLI 命令

```bash
# 单策略回测
af run strategy.py --symbols RB --freq daily --start 2013 --capital 10000000

# 全品种扫描
af scan strategy.py --top 20 --start 2020 --workers 8

# Optuna 参数优化
af optimize strategy.py -s RB,I -f daily -n 200 -o sharpe -j 4

# Walk-Forward 验证
af walkforward strategy.py --symbols RB --train-years 5 --test-years 1

# 品种信息
af info RB

# Paper Trading
af paper strategy.py --symbols RB --freq 1h --data-dir live_data/
af paper-status
af paper-stop

# 实验追踪 (V7.0)
af experiment list
af experiment runs <experiment_name>
af experiment best <experiment_name> -m sharpe
af experiment compare v1 v2 v3
af experiment delete <old_experiment>

# 量化研究 (V7.1)
af purged-cv strategy.py --symbols RB --folds 5 --purge 20 --embargo 5
af wf-optimize strategy.py --symbols RB,I --train-years 5 --test-years 1 --trials 50
af ml-train --symbols RB,I,J --model lgbm --factors returns_5d,volatility_20d --folds 5
```

---

### Iron Rules（引擎硬约束）

1. **信号延迟**：信号在下一个 bar 的 `open_raw` 执行
2. **价格对齐**：成交价 snap 到 `tick_size` 整数倍
3. **FIFO 平仓**：先平昨仓（手续费低），再平今仓
4. **保证金检查**：不足 → 拒绝开仓；`margin_check_mode` 控制频率
5. **涨跌停**：超限 → 重试最多 3 bar
6. **部分成交**：单笔 > `max_fill_ratio` × 成交量 → 部分成交
7. **单方向**：同品种不可同时多空（safe_mode 时静默跳过）
8. **夜盘归属**：夜盘 bars 归属下一交易日
9. **强平**：权益 < 维持保证金 → 宽限期后强制平仓
10. **非线性冲击**：`impact = factor × (lots/vol)^exponent × tick`
11. **买卖价差**：每笔交易额外 `half_spread`
12. **换月成本**：Sigmoid 曲线价差 + 双向手续费
13. **结算价盯市**：每日权益用结算价计算
14. **CFFEX 限额**：IF/IH/IC/IM 每日开仓 ≤ 50 手
15. **锁板拒单**：涨跌停 + 近零成交量 → 拒绝所有订单
16. **止损触发**：SL/TP/Trailing 在 bar 的 high/low 上检查，同 bar 提交，下一 bar open 执行

---

### 常见陷阱

| 陷阱 | 解决 |
|------|------|
| `on_init_arrays` 的 bars 是 dict，不是 BarArray | 用 `context.get_bars()` 或 `context.get_full_close_array()` |
| warmup 不够大 | 设为 >= 最大指标窗口长度（QBase warmup 属性） |
| 5min+ 数据太慢 | QBase 已强制使用 `on_init_arrays` 预计算，符合规范 |
| Optuna 优化日志刷屏 | `suppress_order_logs=True` |
| Industrial 模式 Sharpe 低于 Basic 模式 | 正常，工业级加入了真实成本；衰减 > 50% 才淘汰 |
| MACD histogram 在稳态趋势中 ≈ 0 | 用 MACD line（fast_ema - slow_ema）判断方向，非 histogram |
| 策略参数变异系数 > 0.5 | 参数不稳定，可能过拟合，重新设计或扩大参数范围 |
| `context.bar_index` 在 `on_bar` 中使用 | 正确——QBase 策略通过此索引查 `on_init_arrays` 预计算数组 |
| HTML 报告 K 线图为空 | **必须**传 `bar_data={symbol: bars}` 给 `reporter.generate()`。用 `_load_bars_for_labels()` 加载对应时段的 BarArray，绝不可省略此参数 |
| 报告中看不到指标面板 | 策略必须实现 `get_indicator_panels(datetimes)` 方法；`backtest_runner` 自动注入 metadata |

---

### 指标面板可视化系统

QBase 策略支持在 AlphaForge HTML 报告中渲染指标叠加和副图 panel（类 TradingView 风格）。

#### 架构分工

| 角色 | 负责什么 |
|------|---------|
| QBase（策略层） | 提供数据：从策略对象提取预计算指标数组，分类为 overlay/subplot，打包成 `result.metadata['indicator_panels']` |
| AlphaForge（报告层） | 渲染：用 Plotly make_subplots 画 K 线主图 + 多副图 panel，X 轴联动 zoom |

#### 策略侧实现

每个策略需覆盖 `get_indicator_panels(datetimes)` 方法：

```python
def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
    return {
        "overlays": [
            self._make_overlay("EMA(20)", datetimes, self._ema_fast, color="#ffab40"),
            self._make_overlay("EMA(50)", datetimes, self._ema_slow, color="#ab47bc"),
        ],
        "subplots": [
            self._make_subplot(
                "RSI(14)",
                [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc")],
                horizontal_lines=[30, 70], y_range=[0, 100],
            ),
        ],
    }
```

#### 分类规则

| 指标类型 | 面板分类 | 示例 |
|---------|---------|------|
| 价格级别 | overlay（主图叠加） | EMA, SuperTrend, Bollinger, Donchian, KAMA, HMA, MAMA |
| 振荡器 | subplot（独立副图） | RSI, MACD, ADX, Aroon, CCI, Fisher, STC |
| 量价类 | subplot | CMF, OBV, Force Index, MFI, Chaikin, Klinger |
| 策略信号 | subplot（自动添加） | 由 backtest_runner 自动追加为最后一个 panel |

#### 辅助方法（QBaseStrategy 基类）

```python
# 主图叠加线
_make_overlay(name, datetimes, data, style="line"|"step"|"dash", color=None)

# 副图 panel
_make_subplot(name, traces, height_ratio=0.15, zero_line=False,
              horizontal_lines=None, y_range=None)

# 副图内单条 trace
_make_subplot_trace(name, datetimes, data, style="line"|"bar"|"area"|"step"|"dash",
                    color=None, color_positive=None, color_negative=None)
```

#### 自动注入流程

`backtest_runner.run_qbase_backtest()` 自动处理：
1. 回测完成后调用 `strategy.get_indicator_panels(datetimes)`
2. 自动追加 Signal subplot 作为最后一个 panel
3. 自动为缺少颜色的 trace 分配颜色
4. 写入 `result.metadata['indicator_panels']`
5. AlphaForge `HTMLReportGenerator` 自动读取并渲染

---

### Signal Blending Report

QBase Signal Blending 回测使用 AlphaForge 的 `generate_signal_blend_report()` 生成 hybrid 报告。

#### 调用方式

```python
from alphaforge.report import HTMLReportGenerator

reporter = HTMLReportGenerator()
reporter.generate_signal_blend_report(
    result,                              # blended backtest 结果
    "reports/signal_blend.html",
    bar_data={"I": bars},
    freq="1h",
    strategy_links={"v1": "v1.html"},    # 可选
)
```

#### 数据打包

`scripts/build_portfolio.py` 在 `build_blended_backtest()` 中自动将以下数据打包到 `result.metadata['signal_blend']`：

- `weights` — 各策略权重
- `fdm` — Forecast Diversification Multiplier
- `per_strategy_signals` — 各策略 scaled forecast 数组
- `blended_signal` — 合并后 forecast
- `net_position` — 净仓位（手数）
- `per_strategy_metrics` — 各策略独立回测指标（sharpe, return, dd等）
- `per_strategy_equity` — 各策略归一化 equity 曲线
- `forecast_correlation` — 信号相关性矩阵
- `datetimes` — 时间轴

#### 报告结构

| Section | 说明 |
|---------|------|
| Metrics Cards | 净信号 9 个 KPI |
| Blend Equity Overlay | 净信号粗线 + 各策略细线归一化对比 |
| K-Line | 按策略着色买卖标记 + 净仓位副图（绿红柱+白线） |
| Signal Decomposition | 各策略加权 forecast 堆叠面积图 |
| Weight Pie | 策略权重分配饼图 |
| Drawdown / Monthly Heatmap / Rolling Sharpe | 标准分析图 |
| Forecast Correlation | 信号相关性矩阵热力图 |
| Strategy Comparison | 各策略 vs BLENDED 对比表 |
| Trade Table + Cost Breakdown | 交易明细 |

#### 与单策略报告和 Portfolio 报告的关系

三种报告类型共存，互不影响：

| 报告 | 方法 | 适用场景 |
|------|------|---------|
| 单策略 | `reporter.generate()` | 单个策略回测 |
| Signal Blending | `reporter.generate_signal_blend_report()` | N策略→1净信号→1仓位 |
| Portfolio | `reporter.generate_portfolio_report()` | N策略→N独立仓位 |
