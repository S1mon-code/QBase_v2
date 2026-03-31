# QBase_v2

黑色系中国期货单品种多策略交易系统。

## 核心理念

基本面团队预判市场 regime → 匹配历史同类时段 → 在该类时段上训练策略 → 部署。

## 特性

- **4 Regime 分类：** Strong Trend / Mild Trend / Mean Reversion / Crisis
- **4 维信号体系：** Momentum / Carry / Volume-OI / Technical
- **多周期同步：** 1h / 2h / 4h / daily 同步开发和验证
- **基本面方向约束：** 看多→只做多，看空→只做空
- **Signal Blending：** 多策略信号混合后输出单一净头寸
- **5 维优化函数：** Performance + Significance + Consistency + Risk + Alpha
- **6 层验证：** Regime CV → OOS → Walk-Forward → Deflated Sharpe → Monte Carlo → Industrial
- **5 层归因：** Signal → Horizon → Regime → Baseline Decomposition → Operational

## 品种

黑色系：RB（螺纹钢）、HC（热卷）、I（铁矿石）、J（焦炭）、JM（焦煤）

## 依赖

- **AlphaForge V6.0** — 回测引擎（95 品种，1min-daily，1505 tests）
- **Python 3.10+**
- numpy, numba, optuna, scikit-learn, plotly

## 实现状态

| Phase | 状态 | 完成度 |
|-------|------|--------|
| 1 — 项目骨架 | 完成 | 100% |
| 2 — Regime 标注系统 | 完成 | 90%（缺 visualizer + 真实数据标注）|
| 3 — 风控模块 | 完成 | 100% |
| 4 — 策略开发 | 完成 | 100%（11 strategies）|
| 5 — 优化器 | 完成 | 100% |
| 6 — 验证体系 | 完成 | 100% |
| 7 — 归因分析 | 完成 | 100% |
| 8 — Portfolio 构建 | 完成 | 100% |
| 9 — 扩展品种 | 待实盘数据 | 0% |
| 10 — Pipeline + CLI | 进行中 | 70%（缺 HTML 报告）|
| 11 — 监控 + 实盘 | 进行中 | 60%（缺 paper trading）|

**测试覆盖：** 576+ tests, 100% pass rate

## 已实现策略

| 策略 | Regime | Horizon | 信号维度 |
|------|--------|---------|---------|
| tsmom_fast | trending | fast | momentum |
| tsmom_medium | trending | medium | momentum |
| tsmom_slow | trending | slow | momentum |
| trend_medium_v1 | trending | medium | momentum + volume |
| trend_medium_v2 | trending | medium | momentum + volume |
| trend_medium_v3 | trending | medium | momentum + OI |
| trend_medium_v4 | trending | medium | momentum + volume |
| trend_medium_v5 | trending | medium | momentum + technical |
| trend_fast_v1 | trending | fast | momentum + volume |
| trend_slow_v1 | trending | slow | momentum + technical |
| mr_v1 | mean_reversion | — | technical + momentum |

## 系统架构

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
    │ Layer 8 │  Execution — AlphaForge V6.0 Industrial 模式
    └─────────┘
```

## 快速开始

```bash
# 运行全量测试
python -m pytest tests/ -v

# 标注 regime（需要真实数据）
qbase label RB --visualize

# 运行策略
qbase run trend_medium_v1.py --symbol RB --freq 1h

# 优化
qbase optimize trend_medium_v1.py --symbol RB --freq 1h --regime strong_trend

# 验证（6 层）
qbase validate trend_medium_v1 --all

# Portfolio 构建
qbase portfolio build --symbol RB --regime strong_trend
```

## 文档

- [CLAUDE.md](CLAUDE.md) — Agent 开发指南 + Phase 索引
- [docs/architecture.md](docs/architecture.md) — 系统架构
- [docs/phases/](docs/phases/) — 各 Phase 详细设计
- [todo.md](todo.md) — 任务进度追踪
