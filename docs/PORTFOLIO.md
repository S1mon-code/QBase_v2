# QBase_v2.5 Portfolio 构建指南

> 版本：v2.5 | 更新：2026-04-09
> 状态：**Portfolio 构建完成** — I（long）+ AG（long）已上线
> 前置条件：至少 3 个策略通过完整单策略流程（Phase B-E）

---

## 概览

Portfolio 构建是在多个验证通过的单策略基础上，通过 Signal Blender 合并信号、Regime Allocator 激活策略集，最终输出一个稳健的品种级多策略组合。

```
多个单策略（各自验证通过）
        │
   ┌────▼──────────────┐
   │  策略筛选          │  硬条件过滤，淘汰不合格策略
   └────┬──────────────┘
   ┌────▼──────────────┐
   │  Signal Blender   │  同品种同 Regime 内：多策略信号加权合并
   └────┬──────────────┘
   ┌────▼──────────────┐
   │  Regime Allocator │  基本面预判 → 激活对应策略集（100%）
   └────┬──────────────┘
   ┌────▼──────────────┐
   │  Portfolio 验证    │  LOO + Bootstrap + 评分
   └────┬──────────────┘
   ┌────▼──────────────┐
   │  Holdout 开封      │  最终确认（只跑一次）
   └───────────────────┘
```

---

## 进入 Portfolio 的硬条件

以下条件**全部满足**才能进入 Portfolio：

| 条件 | 阈值 | 检查来源 |
|------|------|---------|
| OOS Sharpe (industrial) | ≥ 0.5 | `validation.yaml` → `industrial.industrial_oos_sharpe` |
| Industrial Sharpe | ≥ 0.5 | `validation.yaml` → `industrial.industrial_sharpe` |
| Industrial 衰减 | ≤ 50% | `validation.yaml` → `industrial.decay_pct` |
| Max Drawdown | ≥ -25% | `validation.yaml` → `max_drawdown` |
| DSR | ≥ 0.95 | `validation.yaml` → `deflated_sharpe` |
| Bootstrap CI 不跨零 | 非 FRAGILE | `validation.yaml` → `bootstrap.verdict` |
| Regime CV | 非 FAIL | `validation.yaml` → `regime_cv.verdict` |
| 独立 Alpha | > 0 | `attribution.md` → Baseline Decomposition |
| 活跃度 | abs(return) > 0.1% | `oos.html` → result metrics |
| 2x成本生存 | Sharpe@2x > 0 | 2倍交易成本下仍盈利 |
| OOS 年化收益 | > 0% | `validation.yaml` → `annualized_return` |
| Profit Factor | ≥ 1.3 | `validation.yaml` → `profit_factor` |
| 单笔最大贡献 | < 30% 总收益 | `validation.yaml` → `max_single_trade_contribution` |

### 交易次数信心分级

交易次数不作为硬筛条件，而是影响权重上限：

| 信心等级 | Daily 交易次数 | 1H 交易次数 | 权重上限 |
|---------|:------------:|:----------:|:-------:|
| HIGH | ≥ 30 | ≥ 100 | 25%（默认上限） |
| MODERATE | 10-29 | 30-99 | 25% |
| LOW | < 10 | < 30 | **15%** |

**原理：** Regime-specific 策略天然交易次数少，不应因此被淘汰。但交易次数少意味着统计信心低，通过限制权重控制风险。

### 组合适配检查（Portfolio Fit）

通过硬筛后，还需通过组合适配检查才能最终入选：

| 检查 | 阈值 | 说明 |
|------|------|------|
| 与现有组合相关性 | < 0.40 | 超过说明冗余，不入选 |
| 边际 Sharpe 贡献 | > 0 | SR_candidate > ρ × SR_portfolio |
| 两两相关性矩阵 | 标记 ≥ 0.40 的对 | 高相关对降权处理 |

**边际 Sharpe 公式：** 新策略的 Sharpe 必须大于 `相关系数 × 现有组合Sharpe`，否则加入后组合Sharpe反而会下降。

### Portfolio 入选标准（从候选池进入 Portfolio）

通过 13 条硬筛的策略进入候选池后，还需满足以下标准才能入选 Portfolio：

#### 硬性要求

| 条件 | 阈值 | 说明 |
|------|------|------|
| OOS Total Return | > 5% | 太低的收益无法覆盖成本 |
| OOS Sharpe | ≥ 1.0 | Portfolio 级别需要更高门槛 |
| 两段 OOS 都盈利 | 是 | 不能只靠一段赚钱 |
| 单段收益占比 | < 70% | 收益不能集中在某一段 |
| 与现有组合相关性 | < 0.5 | 必须带来分散化 |

#### 动态要求

| 条件 | 说明 |
|------|------|
| 加入后 Portfolio Return 提升 | 不拖后腿 |
| 加入后 MaxDD 恶化不超过 3% | 不增加过多风险 |

#### 上限约束

- 最多 **8** 个策略
- Daily 最多 **5** 个，1H 最多 **3** 个
- 同一指标组合不重复

#### 加分项（优先选择）

- 和现有策略用不同指标（信号多样性）
- Daily 和 1H 各至少一个（时间框架分散）
- 独立 Alpha > 20%
- Industrial 衰减 < 20%

完整配置见 `portfolio/long/I/selection_criteria.yaml`

### v2.5 新增：SQS 自动化选择

v2.5 引入两条平行的 Portfolio 构建路径：

| 路径 | 工具 | 适用场景 |
|------|------|---------|
| **Carver Signal Blending** | `portfolio/signal_blender.py` | 手动精调信号合并权重，适合成熟组合 |
| **SQS Portfolio Engine** | `scripts/portfolio_engine.py` | 自动化 SQS 驱动选择，适合快速筛选和扩展品种 |

**SQS 评分**（`scripts/sqs.py`）综合 OOS Sharpe、稳健性、Alpha 独立性等维度打分，Kill Switch（`validation/config.yaml`）自动剔除不合格策略。

**运行入口：** `python scripts/run_portfolio.py`

---

## 两层架构

### Layer 1 — Signal Blender（Carver 标准 Forecast Combination Pipeline）

将多个策略的原始信号（[-1, +1]）通过 Carver/Man AHL 标准 forecast combination pipeline 合并为单一 forecast（[-20, +20]），再统一计算仓位。

#### Forecast Combination Pipeline（6 步）

```
Step 1: Forecast Scaling — 每个策略的原始信号 × forecast_scalar，使 avg|forecast| = 10
Step 2: Forecast Capping — clip 到 ±20
Step 3: Weighted Combination — 加权合并（NaN 策略自动剔除，权重重新归一化）
Step 4: FDM (Forecast Diversification Multiplier) — combined × FDM, 补偿分散化损失
Step 5: Coverage Scaling — 当有信号的策略 < 25% 时，按覆盖率缩放
Step 6: Re-cap — 再次 clip 到 ±20
```

**Forecast Scale 说明：** 策略仍然输出 [-1, +1] 信号，Signal Blender 通过 `forecast_scalar`（基于历史 avg|signal| 校准）将其缩放到 [-20, +20] 范围。10 = 平均信心，20 = 最大信心。

**FDM 公式：**
```
FDM = 1 / sqrt(w' × C × w)
```
其中 `w` 为权重向量，`C` 为 forecast 相关性矩阵。FDM 上限 = 2.5。FDM > 1 说明策略间存在分散化收益，合并后信号强度应放大以补偿。

#### 多频率处理

Signal Blender 在 **1H 频率**上运行：
- 1H 策略：直接使用当前 bar 信号
- Daily 策略：信号 forward-fill 到 1H 网格（每根 1H bar 使用最近的 daily 信号）
- 合并后的 forecast 在 1H 粒度上更新

#### 权重方法（三阶段渐进）

| 阶段 | 方法 | 适用条件 |
|:----:|------|---------|
| 早期 | Equal Weight | 策略数 < 5 |
| 中期 | Inverse Volatility | 策略数 5-10 |
| 成熟期 | HRP × Alpha × Consistency | 策略数 > 10 |

**成熟期权重公式：**
```python
w_hrp = hrp_weights(ledoit_wolf_cov(returns))
alpha_factor = max(0.2, indep_alpha[v] / max_alpha)       # 归因 Layer D
consistency_factor = wf_win_rate[v]                         # 验证 Layer 3
w = normalize(w_hrp × alpha_factor × consistency_factor)
w = clip(w, max_single_weight=0.25)
```

**Horizon 分散约束：** 每个 Horizon（Fast/Medium/Slow）至少 15% 权重。

#### Forecast → Position Sizing

```python
forecast = pipeline_output                                    # [-20, +20]
forecast = directional_filter(forecast, fundamental_view)     # 方向约束
lots = (forecast / 10) × (capital × target_vol) / (price × multiplier × ann_vol)
lots = buffer_zone(lots, current_lots, buffer=0.10)           # 10% position inertia
lots = clip(lots, max_by_margin)                              # 保证金上限
```

**Position Sizing 公式说明：** `forecast/10` 将 forecast 归一化为仓位比例（forecast=10 → 100% 目标仓位，forecast=20 → 200%）。Buffer zone 在目标仓位偏差 < 10% 时不调仓，减少交易频率。

### Layer 2 — Regime Allocator（跨 Regime）

基本面预判确定激活哪个策略集：

| 基本面预判 | 激活策略集 | 资金比例 |
|-----------|----------|---------|
| long | Long 策略集 | 100% |
| short | Short 策略集 | 100% |

**不做 Regime 间混合分配**（基本面团队给确定性预判，不是概率）。

---

## Portfolio 验证

### 进入验证前的准备

1. 对所有通过硬条件的策略调用 `holdout.html` 生成前的最后确认
2. **此时开封 Holdout**，生成每个策略的 `holdout.html`
3. Holdout 结果仅用于确认，不用于修改任何权重

### 验证步骤

**1. Leave-One-Out（LOO）**
去掉任意一个策略，Portfolio Sharpe 不应暴跌（> 20% 下降为 WARNING）

**2. Bootstrap CI**
对 Portfolio 日收益率做 1000 次 Bootstrap，95% CI 下界 > 0

**3. 选择稳定性**
扰动权重 100 次，CORE 策略（始终入选）应 > 50%

**4. Regime 覆盖矩阵**
检查是否所有策略在某个 Regime 下同时亏损（RED FLAG）

### 5 维 15 指标评分（通过标准 ≥ 75 分）

| 维度 | 权重 | 指标 |
|------|:----:|------|
| 收益风险比 | 35% | Sharpe, Calmar, MaxDD, 回撤持续天数, CVaR-95 |
| 信号质量 | 25% | 平均独立 Alpha, Horizon 分散度, vs TSMOM 增量 |
| 组合效率 | 20% | 平均相关性, 回撤重叠率, Portfolio/Best Single Sharpe, 正 Sharpe 策略比例 |
| 稳健性 | 15% | Bootstrap CI 宽度, CORE 占比, Permutation p 均值 |
| 实操性 | 5% | 策略数量, 最大单策略权重, Industrial 平均衰减 |

---

## 报告结构

Portfolio 完成后的报告存放：

```
reports/long/I/
├── portfolio_summary.html    # 主报告（权益叠加 + 相关性矩阵 + 权重饼图）
├── strategy_comparison.html  # 策略对比表
├── coverage_matrix.html      # Regime 覆盖矩阵
├── signal_blending_backtest.html  # Signal Blending Report（由 generate_signal_blend_report() 生成）
├── weights.yaml              # 最终权重
└── validation_summary.yaml  # Portfolio 验证结果
```

---

## 再平衡规则

- **权重更新频率：** 月度（与基本面 review 周期同步）
- **触发条件：** 有新策略加入、策略退役、或新的验证数据
- **不做日级动态调整**（避免过度交易）

---

## 策略退役机制

| 触发条件 | 自动动作 |
|---------|---------|
| 滚动 6 月 Sharpe < 0（持续） | 权重降 50%，标注"观察" |
| 连续 3 个月亏损 | 权重降 50% |
| 滚动 12 月 Sharpe < -0.5 | 移除 Portfolio |
| 实际 MaxDD > 回测 MaxDD × 1.5 | 立即移除 |
| Alpha 衰减检测触发 | 降权 + 调查 |

退役策略代码保留，文件头标注：
```python
# RETIRED: 2026-06-01 原因：滚动12月Sharpe=-0.7，持续亏损
```

---

## 当前状态

> **2026-04-02** — Portfolio 构建完成。
>
> **Signal Blending Pipeline（Carver 标准）：**
> Forecast Scaling (avg|f|=10) → Capping (±20) → 加权合并 → FDM → Re-cap → Direction Filter → Vol-target Sizing
>
> **多频率加法仓位：** `daily_lots + hourly_lots = total`（Daily 和 1H 各自独立 sizing 后加法合并）
>
> **当前 Portfolio 组合：** daily_v37 + 1h_v28
> - Sharpe = 1.979
> - Return = +64.64%
> - MaxDD = 6.36%
> - Calmar = 4.314
> - 策略间实际 correlation = 0.099
>
> **版本重编号说明：** 原 daily_v27 → v37，原 1h_v18 → v28（v2.5 策略从旧 regime 目录迁移至 long/ 后统一重编号）
>
> **Selection 方法：** 使用策略 **returns** 的 correlation，不是 forecast 的 correlation（threshold < 0.5）。Daily策略forecast在1H grid上forward-fill导致forecast correlation失真。
>
> **Risk Allocation：** Daily 70% / 1H 30%
>
> **Report：** 使用 AlphaForge `generate_signal_blend_report()` 生成 signal blending 报告（portfolio.html + 各策略独立报告链接）
>
> 已实现的 Portfolio 工具：
> - `portfolio/signal_blender.py` — Signal Blending
> - `portfolio/weights.py` — EW / IV / HRP+Alpha+Consistency
> - `portfolio/regime_allocator.py` — Regime 激活/休眠
> - `portfolio/selection.py` — 硬条件筛选
> - `portfolio/validation.py` — LOO + Bootstrap + 稳定性
> - `portfolio/scorer.py` — 5 维 15 指标评分

---

## 代码入口

```python
from strategies.long.I.daily.v37 import Strategy as DailyV37
from strategies.long.I.hourly.v28 import Strategy as HourlyV28

from portfolio.signal_blender import SignalBlender
from portfolio.regime_allocator import RegimeAllocator
from portfolio.validation import validate_portfolio
from portfolio.scorer import score_portfolio

# 筛选通过硬条件的策略
eligible = [
    strategies.long.I.daily.v37,
    strategies.long.I.hourly.v28,
]

# 构建 Signal Blender
blender = SignalBlender(eligible, method="equal")   # 早期用 equal
net_signal = blender.blend(signals)

# Portfolio 验证
port_result = validate_portfolio(eligible, blender)
score = score_portfolio(port_result)
print(f"Portfolio score: {score:.1f}/100")
```
