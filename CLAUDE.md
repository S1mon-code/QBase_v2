# QBase_v2 — Agent 开发指南

黑色系中国期货单品种多策略系统。基本面方向约束 + 技术面 regime 适配 + 多周期策略库。

**复用：** indicators/（324 + Carry）+ AlphaForge V7.6.1。其余全部重写。

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
    │         │  Long: TSMOM Baselines + Momentum + Carry + Blended + Technical
    │         │  Short: TSMOM Baselines + Momentum + Carry + Blended + Technical
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
    │ Layer 8 │  Execution — AlphaForge V7.6.1 Industrial 模式
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

**注意：** 策略开发（Phase 4）即在 1h/2h/4h/daily 全周期上同步进行。每个策略从一开始就在多周期上验证。

---

## 核心参数

| 维度 | 规格 |
|------|------|
| 品种 | 黑色系：I → AG → RB → HC → J → JM（大写 ticker） |
| 周期 | 1h / 2h / 4h / daily 同步开发 |
| Regime | Long / Short |
| 信号维度 | Momentum / Carry / Volume-OI / Technical |
| Trend Horizon | Fast(20-60) / Medium(60-125) / Slow(125-250) |
| 方向约束 | 基本面 view: LONG_ONLY / SHORT_ONLY / NEUTRAL |
| 风控 | Chandelier Exit (regime-adaptive) + Vol Targeting + 2% 单笔风险 |
| 优化函数 | 5 维复合: Performance(40%) + Significance(15%) + Consistency(15%) + Risk(15%) + Alpha(15%) |
| 验证 | 6 层: Regime CV → OOS → Walk-Forward → Deflated Sharpe → Monte Carlo → Industrial |
| 归因 | 5 层: Signal → Horizon → Regime → Baseline Decomposition → Operational |
| Portfolio | Signal Blender(同Regime内) + Regime Allocation(跨Regime) |
| 标注方法 | Bry-Boschan 初筛 + 人工校正, ±2月buffer, Long/Short 两类 |
| 切换频率 | 周/月级 |
| 数据分割 | 标注时即分好 train/oos/holdout per regime |

---

## 项目结构

```
QBase_v2/
├── CLAUDE.md                       # 本文件
├── config/
│   ├── settings.yaml               # AlphaForge 路径、全局参数
│   ├── fundamental_views.yaml      # 基本面方向约束
│   └── regime_thresholds.yaml      # Regime 标注阈值
├── data/
│   └── regime_labels/              # 历史 regime 标注 YAML (per instrument)
├── indicators/                     # 324 + Carry 指标
├── regime/                         # Regime 标注 + 匹配
├── strategies/
│   ├── templates/                  # 策略模板 (trending / mean_reversion)
│   ├── baselines/                  # TSMOM Baselines (fast/medium/slow)
│   ├── long/                       # Long regime strategies
│   │   ├── I/{daily,1h,2h,4h}/    # Iron ore long
│   │   └── AG/{daily,1h,2h,4h}/   # Silver long
│   └── short/                      # Short regime strategies
│       ├── I/{daily,1h,2h,4h}/    # Iron ore short
│       └── AG/{daily,1h,2h,4h}/   # Silver short
├── risk/                           # 风控模块
├── optimizer/                      # 优化器
├── validation/                     # 验证体系
├── attribution/                    # 归因分析
├── portfolio/                      # Portfolio 构建
├── pipeline/                       # 流水线编排 + CLI
│   ├── qbase_config.py             # 统一配置中心
│   ├── utils.py                    # Pipeline helper 函数
├── monitoring/                     # 监控 + 实盘
├── reports/                        # HTML 报告
├── research/
│   ├── long/{direction}/{instrument}/{timeframe}/v{N}_{return}%/
│   ├── short/{direction}/{instrument}/{timeframe}/v{N}_{return}%/
│   ├── baselines/I/                # TSMOM baselines
│   └── AG/                         # AG 相关研究
├── research_log/
│   └── trials/                     # 全部试验记录 (Deflated Sharpe 用)
├── scripts/                        # 构建脚本
├── tests/                          # 单元测试 (576+)
├── docs/
│   ├── phases/                     # 各 Phase 详细设计文档
│   └── ALPHAFORGE_API.md           # AlphaForge V7.6.1 完整 API 参考
└── pyproject.toml
```

**路径约定：** 策略路径格式 `strategies/{regime}/{instrument}/{timeframe}/v{N}.py`，其中 regime = long/short。品种用大写 ticker（I, AG, RB, HC, J, JM）。不存在 5min/10min/30min 目录。

---

## 当前实现状态

| Phase | 状态 | 完成度 |
|-------|------|--------|
| 1 — 项目骨架 | ✅ 完成 | 100% |
| 2 — Regime 标注系统 | ✅ 完成 | 100%（I + AG long/short 已标注）|
| 3 — 风控模块 | ✅ 完成 | 100% |
| 4 — 策略开发 | ✅ 完成 | 100%（~270 策略：I + AG，4 timeframe）|
| 5 — 优化器 | ✅ 完成 | 100% |
| 6 — 验证体系 | ✅ 完成 | 100% |
| 7 — 归因分析 | ✅ 完成 | 100% |
| 8 — Portfolio 构建 | ✅ 完成 | 100% |
| 9 — 扩展品种 | ✅ I + AG 完成 | 50%（RB/HC/J/JM 待开发）|
| 10 — Pipeline + CLI | ✅ 完成 | 95%（batch pipeline + 自动报告命名）|
| 11 — 监控 + 实盘 | 🔄 进行中 | 60%（缺 paper trading）|

**测试覆盖：** 576+ tests, 100% pass rate

### 策略规模

| Group | 品种 | 方向 | Regime | Timeframes | 策略数 |
|-------|------|------|--------|-----------|--------|
| long/I | I | long | long | daily/1h/2h/4h | 190 |
| short/I | I | short | short | daily/1h/2h/4h | 40 |
| long/AG | AG | long | long | daily/1h/2h/4h | 40 |
| short/AG | AG | short | short | daily/1h/2h/4h | 40 |

**说明：** Regime 简化为 Long/Short 两类（v2.5 upgrade）。

### Research 文件夹命名

研究结果目录格式：`research/{regime}/{direction}/{instrument}/{timeframe}/v{N}_{+/-}{return}%/`

- return 从 `oos.html` 的「总收益」字段提取，保留两位小数，正数带 `+`
- 示例：`research/long/long/AG/1h/v10_+97.98%/`
- OOS 包含该品种/方向下**所有** `split=oos` 的 regime periods（不按 regime 筛选）
- `run_single_strategy_pipeline()` 自动从 oos.html 提取总收益并命名

### 下一步优先级

1. RB/HC/J/JM 品种扩展
2. Paper Trading 验证
3. 完整 CLI 集成

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

### Holdout 封存

Holdout 数据集只在最终 Portfolio 级别验证时使用一次。策略开发阶段绝不触碰。

### 策略命名

`{regime}_{instrument}_{timeframe}_v{N}` — 如 `long_I_daily_v1`

品种 ticker 大写：I, AG, RB, HC, J, JM

### Git Commit 规范

```
[模块] 类型: 简短描述
示例:
[regime] feat: auto labeler with Bry-Boschan
[strategy] feat: long_I_daily_v15 SuperTrend+VolMom
[optimizer] fix: boundary protection for edge params
```

---

## AlphaForge V7.6.1 集成

AlphaForge 是 QBase_v2 的回测执行引擎（Layer 8）。完整 API 参考见 [docs/ALPHAFORGE_API.md](docs/ALPHAFORGE_API.md)。

**路径**：统一管理在 `pipeline/qbase_config.py`，不在代码中硬编码。

```python
from pipeline.qbase_config import ALPHAFORGE_PATH
import sys
sys.path.insert(0, str(ALPHAFORGE_PATH))

from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.engine.event_driven import EventDrivenBacktester
from alphaforge.engine.config import BacktestConfig
```

**dynamic_margin 修复**：V7.2+ 已修复 dynamic_margin 平仓保证金释放不匹配问题，通过 `PositionEntry.margin_per_lot` 记录开仓时实际保证金率。

### BacktestConfig 推荐配置

```python
# 开发/粗调（快速迭代）
config_dev = BacktestConfig(initial_capital=10_000_000)

# 精调/验证（工业级，QBase 必须）
config_industrial = BacktestConfig(
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

# Optuna 优化（抑制日志）
config_optuna = BacktestConfig(
    initial_capital=10_000_000,
    safe_mode=True,
    suppress_order_logs=True,
)
```

---

## 常见陷阱

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
| HTML 报告 K 线图为空 | **必须**传 `bar_data={symbol: bars}` 给 `reporter.generate()`。用 `_load_bars_for_labels()` 加载对应时段的 BarArray |
| 报告中看不到指标面板 | 策略必须实现 `get_indicator_panels(datetimes)` 方法；`backtest_runner` 自动注入 metadata |
| 4h 交易次数比 1h 高 | 查看"独立交易统计"而非"调仓明细统计"。4h 用 continuous rebalancing 产生大量 partial fills |
| AlphaForge dynamic_margin 平仓保证金释放不匹配 | 已修复：PositionEntry.margin_per_lot 记录开仓时实际保证金率 |
