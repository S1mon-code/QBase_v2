# QBase_v2.5

黑色系中国期货单品种多策略交易系统。

## 核心理念

基本面团队预判市场 regime → 匹配历史同类时段 → 在该类时段上训练策略 → 部署。

## v2.5 新特性

- **Regime 简化：** 4 类 → 2 类 (Long/Short)，Bry-Boschan + R² 过滤，±1 月 buffer
- **SQS 评分：** 策略质量评分系统（`scripts/sqs.py`），自动筛选候选策略
- **Kill Switch：** 自动熔断机制，阈值统一配置于 `validation/config.yaml`
- **Portfolio Engine：** 自动化组合选择（`scripts/portfolio_engine.py`），与 Carver Signal Blending 互补
- **AlphaForge V7.6.1：** NaN 防护 + CFFEX 限额修复 + 保守价差策略
- **rf = 0：** 所有 Sharpe/Sortino 统一使用 rf=0

## 特性

- **2 Regime 分类：** Long / Short（基本面驱动）
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
| 铁矿石 | I | 100 | 已开发（190 策略）|
| 白银 | AG | — | 已开发（80 策略）|
| 螺纹钢 | RB | 10 | 待开发 |
| 热卷 | HC | 10 | 待开发 |
| 焦炭 | J | 100 | 待开发 |
| 焦煤 | JM | 60 | 待开发 |

## 策略规模

| Group | 品种 | 方向 | Regime | 策略数 |
|-------|------|------|--------|--------|
| long/I | I | long | long | 190 |
| short/I | I | short | short | 40 |
| long/AG | AG | long | long | 40 |
| short/AG | AG | short | short | 40 |
| **Total** | | | | **~310** |

## 依赖

- **AlphaForge V7.6.1** — 回测引擎（95 品种，1min-daily，Industrial 模式）
- **Python 3.10+**
- numpy, numba, optuna, scikit-learn, plotly

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
    │ Layer 8 │  Execution — AlphaForge V7.6.1 Industrial 模式
    └─────────┘
```

## 快速开始

```bash
# 运行全量测试
python -m pytest tests/ -v

# 单策略全流程（优化 + 验证 + 归因 + 报告）
python -c "
from pipeline.dev_pipeline import run_single_strategy_pipeline
from strategies.long.I._1h.v1 import LongI1hV1

result = run_single_strategy_pipeline(
    LongI1hV1, symbol='I', direction='long',
    regime='long', horizon='medium', version='v1', freq='1h',
)
# -> research/long/long/I/1h/v1_+96.92%/
"

# 批量运行所有策略
python scripts/batch_optimize_all.py

# SQS 评分（v2.5）
python scripts/sqs.py

# Portfolio Engine（v2.5）
python scripts/run_portfolio.py

# CLI
qbase label I --visualize
qbase run v1.py --symbol I --freq 1h
qbase optimize v1.py --symbol I --regime long
qbase validate v1 --all
qbase portfolio build --symbol I --regime long
```

## 文档

- [CLAUDE.md](CLAUDE.md) — Agent 开发指南
- [docs/ALPHAFORGE_API.md](docs/ALPHAFORGE_API.md) — AlphaForge V7.6.1 API 参考
- [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) — 单策略开发标准流程 v3.0
- [docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md) — 策略开发指引
- [docs/PORTFOLIO.md](docs/PORTFOLIO.md) — Portfolio 构建标准
- [docs/architecture.md](docs/architecture.md) — 系统架构
- [docs/phases/](docs/phases/) — 各 Phase 详细设计
