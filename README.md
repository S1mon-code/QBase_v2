# QBase_v2

黑色系中国期货单品种多策略交易系统。

## 核心理念

基本面团队预判市场 regime → 匹配历史同类时段 → 在该类时段上训练策略 → 部署。

## 特性

- **4 Regime 分类：** Strong Trend / Mild Trend / Mean Reversion / Crisis
- **4 维信号体系：** Momentum / Carry / Volume-OI / Technical
- **多周期同步：** 1h / 2h / 4h / daily 同步开发和验证
- **基本面方向约束：** 看多→只做多，看空→只做空
- **Signal Blending：** 多策略信号混合后输出单一净头寸（Carver 标准）
- **5 维优化函数：** Performance + Significance + Consistency + Risk + Alpha
- **6 层验证：** Regime CV → OOS → Walk-Forward → Deflated Sharpe → Monte Carlo → Industrial
- **5 层归因：** Signal → Horizon → Regime → Baseline Decomposition → Operational
- **自动报告命名：** Research 文件夹以 OOS 总收益自动命名（如 `v10_+97.98%`）

## 品种

| 品种 | 代码 | 乘数 | 状态 |
|------|------|------|------|
| 铁矿石 | I | 100 | 已开发（270 策略）|
| 白银 | AG | — | 已开发（80 策略）|
| 螺纹钢 | RB | 10 | 待开发 |
| 热卷 | HC | 10 | 待开发 |
| 焦炭 | J | 100 | 待开发 |
| 焦煤 | JM | 60 | 待开发 |

## 依赖

- **AlphaForge V7.1** — 回测引擎（95 品种，1min-daily，Industrial 模式）
- **Python 3.10+**
- numpy, numba, optuna, scikit-learn, plotly

## 实现状态

| Phase | 状态 | 完成度 |
|-------|------|--------|
| 1 — 项目骨架 | 完成 | 100% |
| 2 — Regime 标注系统 | 完成 | 100%（I + AG long/short）|
| 3 — 风控模块 | 完成 | 100% |
| 4 — 策略开发 | 完成 | 270 策略（I + AG，4 timeframe）|
| 5 — 优化器 | 完成 | 100% |
| 6 — 验证体系 | 完成 | 100% |
| 7 — 归因分析 | 完成 | 100% |
| 8 — Portfolio 构建 | 完成 | 100% |
| 9 — 扩展品种 | I + AG 完成 | 50% |
| 10 — Pipeline + CLI | 完成 | 95%（batch pipeline + 自动报告命名）|
| 11 — 监控 + 实盘 | 进行中 | 60%（缺 paper trading）|

**测试覆盖：** 576+ tests, 100% pass rate

## 策略规模

| Group | 策略数 | 通过验证 | Gate Fail |
|-------|--------|---------|-----------|
| strong_trend/long/I | 110 | 102 | 8 |
| strong_trend/long/AG | 40 | 35 | 5 |
| strong_trend/short/AG | 40 | 27 | 13 |
| mild_trend/long/I | 40 | 待标注 | — |
| mild_trend/short/I | 40 | 18 | 22 |
| **Total** | **270** | **182** | **48** |

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

## 快速开始

```bash
# 运行全量测试
python -m pytest tests/ -v

# 单策略全流程（优化 + 验证 + 归因 + 报告）
python -c "
from pipeline.dev_pipeline import run_single_strategy_pipeline
from strategies.strong_trend.long.AG._1h.v1 import StrongTrendLongAG1hV1

result = run_single_strategy_pipeline(
    StrongTrendLongAG1hV1, symbol='AG', direction='long',
    regime='strong_trend', horizon='medium', version='v1', freq='1h',
)
# -> research/strong_trend/long/AG/1h/v1_+96.92%/
"

# 批量运行所有策略
python scripts/batch_optimize_all.py

# CLI
qbase label AG --visualize
qbase run v1.py --symbol AG --freq 1h
qbase optimize v1.py --symbol AG --regime strong_trend
qbase validate v1 --all
qbase portfolio build --symbol AG --regime strong_trend
```

## 文档

- [CLAUDE.md](CLAUDE.md) — Agent 开发指南 + AlphaForge API 参考
- [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) — 单策略开发标准流程 v3.0
- [docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md) — 策略开发指引
- [docs/PORTFOLIO.md](docs/PORTFOLIO.md) — Portfolio 构建标准
- [docs/architecture.md](docs/architecture.md) — 系统架构
- [docs/phases/](docs/phases/) — 各 Phase 详细设计
