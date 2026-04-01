# QBase_v2 单策略开发标准流程 v3.0

> 版本：v3.0 | 更新：2026-04-01
> 适用范围：单策略从零到验证完成。Portfolio 构建见 [PORTFOLIO.md](PORTFOLIO.md)。
> 详细规格：Phase 4-7 详见 [phases/](phases/) 目录，本文档为流程总览，不重复细节。

---

## 文件分类体系

```
QBase_v2/
│
├── strategies/                                # 【代码库】策略源码
│   ├── templates/                             # 基类模板（不动）
│   │   ├── base_strategy.py
│   │   ├── trending_template.py
│   │   └── mean_reversion_template.py
│   │
│   ├── baselines/                             # TSMOM 三档 Baseline（不动）
│   │   ├── tsmom_fast.py
│   │   ├── tsmom_medium.py
│   │   └── tsmom_slow.py
│   │
│   ├── strong_trend/                          # Regime: 强趋势
│   │   ├── long/                              # 方向: 只做多
│   │   │   ├── iron/                          # 品种: 铁矿石
│   │   │   │   ├── daily/                     # Timeframe
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── v1.py
│   │   │   │   │   ├── v2.py
│   │   │   │   │   └── ...
│   │   │   │   ├── 1h/
│   │   │   │   ├── 2h/
│   │   │   │   ├── 4h/
│   │   │   │   ├── 30min/
│   │   │   │   ├── 10min/
│   │   │   │   └── 5min/
│   │   │   ├── ag/                            # 品种: 白银
│   │   │   └── lc/                            # 品种: ...
│   │   └── short/                             # 方向: 只做空
│   │       ├── iron/
│   │       └── ...
│   │
│   ├── mild_trend/                            # Regime: 温和趋势
│   │   ├── long/
│   │   └── short/
│   │
│   ├── mean_reversion/                        # Regime: 均值回归（无方向分层，双向交易）
│   │   ├── iron/
│   │   │   ├── daily/
│   │   │   ├── 1h/
│   │   │   └── ...
│   │   └── ag/
│   │
│   └── crisis/                                # Regime: 危机
│       ├── long/
│       └── short/
│
├── research/                                  # 【研究产出】镜像 strategies/ 结构
│   ├── strong_trend/
│   │   ├── long/
│   │   │   ├── iron/
│   │   │   │   ├── daily/
│   │   │   │   │   ├── v1/
│   │   │   │   │   │   ├── params.yaml        # 优化参数
│   │   │   │   │   │   ├── validation.yaml    # 验证结果
│   │   │   │   │   │   ├── attribution.md     # 归因分析
│   │   │   │   │   │   ├── train.html         # AlphaForge IS 回测报告
│   │   │   │   │   │   ├── oos.html           # AlphaForge OOS 回测报告
│   │   │   │   │   │   └── holdout.html       # Holdout 报告（开封后）
│   │   │   │   │   ├── v2/
│   │   │   │   │   └── summary.yaml           # 该 timeframe 所有版本汇总
│   │   │   │   └── 1h/
│   │   │   └── ag/
│   │   └── short/
│   ├── mild_trend/
│   ├── mean_reversion/                        # 无 direction 层
│   └── crisis/
│
├── research_log/
│   └── trials/
│       └── trial_registry.jsonl               # 全部 Optuna trial（不可删除）
│
└── data/
    └── regime_labels/
        └── {symbol}.yaml                      # Regime 标注
```

**层级规则：** `{regime}/{direction}/{instrument}/{timeframe}/v{N}.py`

**Mean Reversion 例外：** `mean_reversion/{instrument}/{timeframe}/v{N}.py`（无方向分层，MR 策略天然双向交易）

**策略 name 属性：** `name = "{regime}_{direction}_{instrument}_{timeframe}_v{N}"`

示例：
- `"strong_trend_long_iron_daily_v1"`
- `"strong_trend_long_iron_1h_v5"`
- `"mean_reversion_iron_daily_v3"`（MR 无方向前缀）

---

## 完整开发流程

### Phase A: 前置准备（每个新品种/新方向做一次）

#### A1. 基本面确认

更新 `config/fundamental_views.yaml`：

```yaml
views:
  I:
    direction: long          # long / short / neutral
    regime: strong_trend     # strong_trend / mild_trend / mean_reversion / crisis
updated_at: "2026-04-01"
updated_by: "simon - I iron ore bullish"
```

**原则：** direction + regime 确定后锁定，不随开发过程改变。

#### A2. Regime 标注

1. 运行 Bry-Boschan 自动初标
2. 人工校正（添加 `driver` 经济逻辑描述）
3. **标注时同步完成 train/oos/holdout 切分（60/20/20，按时间顺序）**
4. 写入 `data/regime_labels/{symbol}.yaml`
5. **Holdout 从此刻起封存，不得查看，直到最终 Portfolio 确认时才开封**

标注格式：
```yaml
- start: "2015-06-01"
  end: "2016-02-28"
  regime: strong_trend
  direction: up
  driver: "供给侧改革推动价格上涨"
  buffer_start: "2015-04-01"
  buffer_end: "2016-04-30"
  split: train              # train / oos / holdout
```

#### A3. 建立 TSMOM Baseline

```python
from pipeline.dev_pipeline import run_baselines

baselines = run_baselines("I", "long", "strong_trend", freq="daily")
# 输出: {"fast": 0.42, "medium": 0.58, "slow": 0.31}
# 保存: research/strong_trend/long/iron/daily/baselines/{fast,medium,slow}.html
```

**这三个 Sharpe 是后续所有策略的最低门槛。**

---

### Phase B: 单策略编写（每个策略重复）

> 详见 [phase04-strategy-development.md](phases/phase04-strategy-development.md)

#### B1. 选指标

从 `indicators/` 库中选 1-3 个指标：

| 信号维度 | 适用 Regime | 代表指标 |
|---------|------------|---------|
| Momentum | 所有趋势 | MACD, EMA, ADX, Vortex, Aroon |
| Volume-OI | 趋势确认 | CMF, Klinger, OBV, Force Index |
| Technical | 辅助确认 | Bollinger Bands, Keltner, ATR |
| Carry | 期限结构 | 近远月价差, Roll Yield |

**规则：**
- Trending 策略**必须**含 Momentum 维度
- 总参数 <= 5 个（含 `chandelier_mult`）
- 指标选择有逻辑依据（与目标 Regime 特征匹配）

#### B2. 编写策略代码

```python
# strategies/strong_trend/long/iron/daily/v1.py
from strategies.templates.base_strategy import QBaseStrategy

class StrongTrendLongIronDailyV1(QBaseStrategy):
    name: ClassVar[str] = "strong_trend_long_iron_daily_v1"
    regime: ClassVar[str] = "trending"
    horizon: ClassVar[str] = "medium"
    signal_dimensions: ClassVar[list[str]] = ["momentum", "volume"]
    warmup: ClassVar[int] = 75

    # 可优化参数（<= 5 个，含 chandelier_mult）
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        # 预计算指标（必须在此完成，禁止在 _generate_signal 中计算）
        self._macd_line, _, _ = macd(self._closes, self.fast_period, self.slow_period, 9)

    def _generate_signal(self, bar_index: int) -> float:
        """long/ 目录: 返回 [0, +1]，只做多"""
        if bar_index < self.warmup:
            return 0.0
        # ... 信号逻辑
        return signal

    def get_indicator_config(self) -> list[dict]:
        return [{"name": "MACD", "params": {"fast": self.fast_period, "slow": self.slow_period}}]
```

**方向约束：**
- `long/` 目录：信号范围 `[0, 1]`，只做多
- `short/` 目录：信号范围 `[-1, 0]`，只做空
- `mean_reversion/` 目录：信号范围 `[-1, 1]`，双向交易

#### B3. 裸逻辑门控

使用默认参数，在 train 集全历史上验证：

| 指标 | 门槛 | 不通过 |
|------|------|--------|
| 交易次数 | >= 10（daily）/ >= 30（1h） | 直接丢弃 |
| **年化收益** | **> 0** | **直接丢弃** |
| Profit Factor | **>= 1.2** | 直接丢弃 |
| 权益曲线 | 大致向上 | 重新设计 |

**不通过裸逻辑门控 -> 不进入优化器，节省时间。**

#### B4. vs TSMOM Baseline 对比（软检查）

- 裸逻辑跑出的 Sharpe vs 对应 Horizon 的 TSMOM Baseline
- 低于 Baseline -> 警告（软标注，不强制丢弃，但要注意）

#### B5. Regime 时段预验证（软检查）

- 只在 train 集目标 regime 时段跑的 Sharpe 应 > 全历史 Sharpe
- 否则说明策略信号与 Regime 不匹配

---

### Phase C: 参数优化（5 维两阶段）

> 详见 [phase05-optimizer.md](phases/phase05-optimizer.md)

**只在 train 集的目标 regime 时段上优化（含 buffer）**

#### 两阶段 Optuna

```
Phase 1 粗调：30 trials，全范围 TPE，S_performance 用 tanh（防追极端 Sharpe）
Phase 2 精调：50 trials，最优 +/-15% 范围，S_performance 用 linear
```

**精调阶段使用 AlphaForge Industrial 模式，确保在真实成本下优化参数。**

**训练和验证都使用 active_periods mask，策略只在目标regime段内交易，非regime段信号归零，权益曲线保持平坦。**

#### 5 维复合目标函数

```
score = 0.40 * S_performance    # 0.6*sharpe + 0.4*return（非纯 Sharpe）
       + 0.15 * S_significance  # Lo (2002) t-stat 统计显著性
       + 0.15 * S_consistency   # 分窗口 Sharpe 一致性
       + 0.15 * S_risk          # MaxDD + CVaR-95 尾部风险
       + 0.15 * S_alpha         # vs TSMOM Baseline 超额（<=0 硬过滤）
```

**S_performance 混合公式：** `S_performance = 0.6 * sharpe + 0.4 * return`，确保优化器同时追求风险调整后收益和绝对收益，避免高 Sharpe / 低收益陷阱。

**硬过滤：**
- 交易次数不足 -> 返回 -10（死区）
- S_alpha <= 0（不如 Baseline）-> 返回 -5

#### 稳健性检查

最优参数 +/-15% 邻域采样 20 个点：
- >= 60% 邻居 > 最优 50% -> **PLATEAU**（稳定，通过）
- 否则 -> **SPIKE**（参数不稳定，软标注，谨慎使用）

#### Trial 记录

每次 trial 自动写入 `research_log/trials/trial_registry.jsonl`，**不可删除**（Deflated Sharpe 计算依赖完整记录）。

优化结果保存：`research/{regime}/{direction}/{instrument}/{timeframe}/v{N}_{+/-}{return}%/params.yaml`

---

### Phase D: 6 层验证（顺序执行）

> 详见 [phase06-validation.md](phases/phase06-validation.md)

**验证使用已锁定的优化参数，禁止因验证结果修改参数。**

| 层 | 名称 | 方法 | 硬淘汰条件 | 软标注 |
|----|------|------|-----------|--------|
| L1 | Regime CV | 全段连续回测 + LOO：每次mask掉一个regime段，跑完整训练集，计算剩余段的Sharpe | Mean Sharpe <= 0 且 Win Rate < 33% -> FAIL | MARGINAL |
| L2 | OOS 验证 | WF Ratio = OOS_Sharpe / IS_Sharpe | **OOS 年化收益 <= 0 -> FAIL** | WF Ratio < 0.5，行为异常 |
| L3 | Walk-Forward | Regime-Aware（每个 regime 时段作窗口） | — | Win Rate < 50% |
| L4 | Deflated Sharpe | Bailey & Lopez de Prado (2014) 多重检验校正 | — | DSR < 0.95 |
| L5a | Bootstrap | 1000 次有放回重采样 CI | CI 跨零 -> FRAGILE | — |
| L5b | Permutation | 1000 次打乱收益序列 | — | p-value > 0.10 |
| L6a | Industrial | AlphaForge Industrial 模式对比 | 衰减 > 50% | 衰减 30-50% |
| L6b | Stress | 2x 滑点 Sharpe 衰减 | — | 衰减 > 30% |

**v3.0 新增的绝对收益检查：**
- L2 OOS 阶段：年化收益 <= 0 直接 FAIL
- L6a Industrial 阶段：Profit Factor < 1.3 -> FAIL
- **高 Sharpe / 低收益 WARNING：** Sharpe >= 1.0 但年化收益 < 5% -> 软标注 `HIGH_SHARPE_LOW_RETURN`

**validation.yaml 现在包含 oos_period_breakdown（分段明细）和 period_concentration_warning（单段贡献 > 70% 告警）。**

验证结果保存：`research/{regime}/{direction}/{instrument}/{timeframe}/v{N}_{+/-}{return}%/validation.yaml`

---

### Phase E: 5 层归因（单策略完结前必做）

> 详见 [phase07-attribution.md](phases/phase07-attribution.md)

归因结果影响未来 Portfolio 权重分配，现在记录，以后用。

| 层 | 内容 | 关键输出 |
|----|------|---------|
| A | Signal Attribution（Shapley <= 4 信号 / Ablation > 4） | 哪个指标是主力，哪个 < 5% 是冗余 |
| B | Horizon Fingerprint（TSMOM 1M/3M/12M OLS 回归） | Alpha 属于哪个周期，R-squared |
| C | Regime Attribution（按标注时段分类交易） | 在哪些 Regime 下赚/亏钱 |
| D | Baseline Decomposition（TSMOM + Carry 两因子 OLS） | 独立 Alpha 占比（Portfolio 权重核心依据） |
| E | Operational（Basic vs Industrial Sharpe 成本分解） | 各成本项对 Sharpe 的侵蚀 |

归因报告保存：`research/{regime}/{direction}/{instrument}/{timeframe}/v{N}_{+/-}{return}%/attribution.md`

---

### Phase F: AlphaForge 报告生成（必须）

**每个通过验证的策略都必须生成 AlphaForge 报告。**

#### Train 报告（train.html）

```python
from alphaforge.engine.config import BacktestConfig
from alphaforge.engine.event_driven import EventDrivenBacktester

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

engine = EventDrivenBacktester(spec_manager=specs, config=config)
result = engine.run(strategy, {symbol: train_bars})

# 生成 HTML 报告
reporter = engine.create_reporter()
reporter.generate(
    result,
    output_path="research/strong_trend/long/iron/daily/v1_+15.89%/train.html",
    bar_data={symbol: train_bars},  # 必须传入，否则 K 线图为空
)
```

#### OOS 报告（oos.html）

同上流程，使用 OOS 期间数据。

#### Holdout 报告（holdout.html）

**仅在 Portfolio 最终确认时生成，一次性开封。** 不可提前查看。

#### 报告必须包含的内容

- 权益曲线 + 基准对比
- K 线图 + 信号标注
- 交易明细表
- 核心指标：Sharpe, Calmar, MaxDD, Win Rate, Profit Factor
- 衰减检测（DecayDetector）：stability_score, rolling_sharpe_slope

**指标面板**：`run_qbase_backtest()` 自动调用 `strategy.get_indicator_panels()` 并注入到 `result.metadata['indicator_panels']`，无需手动处理。AlphaForge `HTMLReportGenerator.generate()` 自动渲染。

---

## 单策略完结标准

以下全部满足，才视为单策略开发完成：

- [ ] 裸逻辑门控 PASS（含年化收益 > 0、PF >= 1.2）
- [ ] 优化完成，`params.yaml` 已保存
- [ ] 稳健性 PLATEAU（或 SPIKE 已记录并接受）
- [ ] 6 层验证无硬淘汰
- [ ] OOS 年化收益 > 0 确认
- [ ] Industrial Profit Factor >= 1.3
- [ ] 单笔集中度检查通过（top1 trade < 30% total P&L）
- [ ] `train.html` + `oos.html` 已生成（Industrial 模式，必须传 `bar_data`）
- [ ] `attribution.md` 已生成
- [ ] Holdout 仍处于封存状态
- [ ] `summary.yaml` 已更新

---

## 准入标准汇总

### 硬筛条件（13 条，全部满足才能入选 Portfolio）

| # | 条件 | 阈值 | 检查来源 |
|---|------|------|---------|
| 1 | OOS 年化收益 | > 0% | `validation.yaml -> oos_annualized_return` |
| 2 | OOS Sharpe (industrial) | >= 0.5 | `validation.yaml -> industrial_oos_sharpe` |
| 3 | Industrial Sharpe | >= 0.5 | `validation.yaml -> industrial.industrial_sharpe` |
| 4 | Industrial 衰减 | <= 50% | `validation.yaml -> industrial.decay_pct` |
| 5 | Max Drawdown | >= -25% | `validation.yaml -> max_drawdown` |
| 6 | DSR | >= 0.95 | `validation.yaml -> deflated_sharpe` |
| 7 | Bootstrap CI | 不跨零（非 FRAGILE） | `validation.yaml -> bootstrap.verdict` |
| 8 | Regime CV | 非 FAIL | `validation.yaml -> regime_cv.verdict` |
| 9 | 独立 Alpha | > 0 | `attribution.md -> Baseline Decomposition` |
| 10 | 活跃度 | abs(return) > 0.1% | `oos.html -> result metrics` |
| 11 | 2x 成本生存 | Sharpe@2x > 0 | 2 倍交易成本下仍盈利 |
| 12 | Profit Factor | >= 1.3 | `validation.yaml -> profit_factor` |
| 13 | 单笔集中度 | top1 trade < 30% total P&L | `oos.html -> trade analysis` |

### 交易次数信心分级（影响权重上限）

交易次数不作为硬筛，而是影响权重上限：

| 信心等级 | Daily 交易次数 | 1H 交易次数 | 权重上限 |
|---------|:------------:|:----------:|:-------:|
| HIGH | >= 30 | >= 100 | 25% |
| MODERATE | 10-29 | 30-99 | 25% |
| LOW | < 10 | < 30 | **15%** |

### 组合适配检查（Portfolio Fit）

| 检查 | 阈值 | 说明 |
|------|------|------|
| 与现有组合相关性 | < 0.40 | 超过则不入选 |
| 边际 Sharpe 贡献 | > 0 | SR_candidate > rho * SR_portfolio |
| 两两相关性矩阵 | 标记 >= 0.40 的对 | 高相关对降权 |

---

## 关键约束

| 约束 | 说明 |
|------|------|
| 预计算必须 | 所有指标必须在 `on_init_arrays` 中预计算，`_generate_signal` 只做查表 |
| 参数上限 | 每策略可优化参数 <= 5 个（含 `chandelier_mult`） |
| 测试集只读 | OOS/Holdout 结果不能用于修改参数 |
| Trial 不可删 | `research_log/trials/trial_registry.jsonl` 是 Deflated Sharpe 的计算基础 |
| Holdout 封存 | 开发期间不得查看 Holdout，只在 Portfolio 最终确认时开封 |
| Baseline 先行 | 每个品种/方向必须先建立 TSMOM Baseline，再开发策略 |
| Industrial 必须 | AlphaForge 报告必须使用 Industrial 模式，必须传 `bar_data` |
| 绝对收益门控 | 裸逻辑年化收益 > 0，OOS 年化收益 > 0，PF >= 1.3 |
| 单笔集中度 | 单笔交易最大贡献 < 30% 总 P&L |
| Carver 连续调仓 | 仓位管理使用 Carver 连续调仓方式：每根 bar 重新计算目标仓位，偏差 > 10% 时自动加仓或减仓。SIGNAL_THRESHOLD = 0.05, REBALANCE_BUFFER = 10% |
| Position Sizing 频率规则 | **Daily**：连续调仓 + 10% buffer（每 bar 重算，偏差 > 10% 才调仓）。**1H 及更快频率**：固定入场 sizing（入场时计算一次，不持续调仓）。**Portfolio 层面**：加法合并 `daily_lots + hourly_lots = total` |
| 指标面板必须 | 所有策略必须实现 `get_indicator_panels()` 方法，提供指标面板可视化数据 |
| Signal Blender Pipeline | Signal Blender 使用 Carver 标准 forecast combination pipeline: Forecast Scaling (avg\|f\|=10) → Capping (±20) → 加权合并 → FDM → Re-cap → Direction Filter → Vol-target Sizing。策略输出 [-1,+1]，Blender 缩放到 [-20,+20]。多频率在 1H 网格上合并（daily 信号 forward-fill） |
| Timeframe 杠铃 | 推荐 Daily(55%) + 1H(45%) 核心组合，4H/2H 作为可选扩展 |
| MR 无方向层 | Mean Reversion 策略天然双向，目录中无 long/short 分层 |
| 高 Sharpe 低收益预警 | Sharpe >= 1.0 但年化收益 < 5% 必须标注 WARNING |
| OOS 全 regime | OOS 包含该品种/方向下**所有** `split=oos` 的 regime periods，不按 regime 类型筛选 |
| Research 文件夹命名 | `v{N}_{+/-}{return}%`，return 从 `oos.html` 的「总收益」字段提取，保留两位小数，正数带 `+` |
| Portfolio入选标准 | OOS Return>5%, Sharpe≥1.0, 两段OOS盈利, return correlation<0.5（不用forecast correlation）, 最多8个策略 | portfolio/selection_criteria.yaml |

---

## 自动化调用

完整流水线一键运行：

```python
from pipeline.dev_pipeline import run_baselines, run_single_strategy_pipeline
from strategies.strong_trend.long.iron.daily.v1 import StrongTrendLongIronDailyV1

# Phase A3: 建立 Baseline（每个品种/方向只需跑一次）
baselines = run_baselines("I", "long", "strong_trend", freq="daily")

# Phase B-F: 全流程（含优化 + 验证 + 归因 + AlphaForge 报告）
result = run_single_strategy_pipeline(
    StrongTrendLongIronDailyV1,
    symbol="I",
    direction="long",
    regime="strong_trend",
    timeframe="daily",
    version="v1",
)

print(result["validation"]["oos_sharpe"])          # OOS Sharpe
print(result["validation"]["oos_annualized_return"])  # OOS 年化收益
print(result["validation"]["hard_reject"])          # 是否硬淘汰
print(result["validation"]["profit_factor"])        # Profit Factor
print(result["output_dir"])                         # 输出路径
# -> research/strong_trend/long/iron/daily/v1_+15.89%/
```

跳过优化（使用已知参数）：

```python
result = run_single_strategy_pipeline(
    StrongTrendLongIronDailyV1,
    symbol="I",
    direction="long",
    regime="strong_trend",
    timeframe="daily",
    version="v1",
    params_override={"fast_period": 12, "slow_period": 26, "chandelier_mult": 2.5},
)
```

---

## 扩展品种流程

顺序：RB -> HC -> I -> J -> JM

每个新品种重复 Phase A-F：
- **策略代码不变**，只重新优化参数（在新品种的 regime 时段上）
- 品种级阈值可能不同（铁矿 strong_trend_pct=0.25，焦煤=0.15）
- 单品种 regime 时段不够时，允许同板块品种辅助训练，但 OOS/Holdout 仍只用目标品种
