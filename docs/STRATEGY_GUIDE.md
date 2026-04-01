# QBase_v2 策略开发指引

> 本文档是快速参考。完整流程见 [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)

> 版本：v2.0 | 更新：2026-04-01
> 本文档是所有策略开发、研究、验证、报告的唯一权威指引。

---

## 1. 目录架构

### 1.1 strategies/ — 策略代码

```
strategies/
├── templates/                          # 基类模板（不动）
│   ├── base_strategy.py
│   ├── trending_template.py
│   └── mean_reversion_template.py
│
├── baselines/                          # TSMOM Baselines（不动）
│   ├── tsmom_fast.py
│   ├── tsmom_medium.py
│   └── tsmom_slow.py
│
├── strong_trend/                       # Regime: 强趋势
│   ├── long/                           # 方向: 只做多
│   │   ├── iron/                       # 品种: 铁矿石
│   │   │   ├── daily/                  # Timeframe
│   │   │   │   ├── __init__.py
│   │   │   │   ├── v1.py
│   │   │   │   ├── v2.py
│   │   │   │   └── ...
│   │   │   ├── 1h/
│   │   │   ├── 2h/
│   │   │   ├── 4h/
│   │   │   ├── 30min/
│   │   │   ├── 10min/
│   │   │   └── 5min/
│   │   ├── ag/                         # 品种: 白银
│   │   └── lc/                         # 品种: ...
│   └── short/                          # 方向: 只做空
│       ├── iron/
│       └── ...
│
├── mild_trend/                         # Regime: 温和趋势
│   ├── long/
│   └── short/
│
├── mean_reversion/                     # Regime: 均值回归（无方向分层，双向交易）
│   ├── iron/
│   │   ├── daily/
│   │   ├── 1h/
│   │   └── ...
│   └── ag/
│
└── crisis/                             # Regime: 危机
    ├── long/
    └── short/
```

**层级规则：** `regime / direction / instrument / timeframe / v{N}.py`

**Mean Reversion 例外：** `mean_reversion / instrument / timeframe / v{N}.py`（无方向分层，因为 MR 策略天然双向交易）

### 1.2 research/ — 研究结果（镜像 strategies/）

```
research/
├── strong_trend/
│   ├── long/
│   │   ├── iron/
│   │   │   ├── daily/
│   │   │   │   ├── v1/
│   │   │   │   │   ├── params.yaml         # 优化参数
│   │   │   │   │   ├── validation.yaml      # 验证结果
│   │   │   │   │   ├── attribution.md       # 归因分析
│   │   │   │   │   ├── train.html           # AlphaForge IS 回测报告
│   │   │   │   │   ├── oos.html             # AlphaForge OOS 回测报告
│   │   │   │   │   └── holdout.html         # AlphaForge Holdout 报告（开封后）
│   │   │   │   ├── v2/
│   │   │   │   └── summary.yaml             # 该 timeframe 所有版本汇总
│   │   │   └── 1h/
│   │   └── ag/
│   └── short/
├── mild_trend/
├── mean_reversion/
└── crisis/
```

### 1.3 reports/ — Portfolio 级报告

```
reports/
├── strong_trend/
│   ├── long/
│   │   └── iron/
│   │       ├── portfolio_summary.html       # 主报告（权益曲线 + 相关性矩阵 + 权重饼图）
│   │       ├── strategy_comparison.html     # 策略对比表
│   │       ├── coverage_matrix.html         # Regime 覆盖矩阵
│   │       ├── weights.yaml                 # 最终权重
│   │       └── validation_summary.yaml      # Portfolio 验证结果
│   └── short/
│       └── iron/
├── mild_trend/
├── mean_reversion/
└── crisis/
```

---

## 2. 策略命名规范

### 2.1 文件命名

```
v{N}.py
```

- N 从 1 开始递增，**同一个 timeframe 目录内唯一**
- 不同 timeframe 下的 v1.py 是**完全独立的策略**，允许代码相同或不同

### 2.2 策略 name 属性

```python
name = "{regime}_{direction}_{instrument}_{timeframe}_v{N}"
```

示例：
- `"strong_trend_long_iron_daily_v1"`
- `"strong_trend_long_iron_1h_v5"`
- `"mean_reversion_iron_daily_v3"`

### 2.3 研究结果目录

```
research/{regime}/{direction}/{instrument}/{timeframe}/v{N}_{+/-}{return}%/
```

文件夹名包含 OOS 总收益（从 `oos.html` 的「总收益」字段提取），格式：`v{N}_{+/-}{return}%`（保留两位小数，正数带 `+`）。

示例：
- `research/strong_trend/long/AG/1h/v10_+97.98%/`
- `research/strong_trend/long/iron/daily/v1_+15.89%/`

`run_single_strategy_pipeline()` 会自动从 `oos.html` 提取总收益并命名文件夹。

**OOS 范围：** OOS 包含该品种/方向下所有 `split=oos` 的 regime periods（不限制 regime 类型），以反映策略在全市场条件下的表现。

---

## 3. 策略开发流程

### 3.1 完整流程（每个策略必须走完）

```
Step 1: 设计策略
  ↓ 选择信号维度组合，确定指标，写代码
Step 2: 预计算验证
  ↓ 确保 on_init_arrays + _generate_signal 正确
Step 3: 优化（Optimizer）
  ↓ Regime-specific 两阶段 Optuna 优化
  ↓ 精调阶段(Fine phase)使用 Industrial 模式，优化器使用 active_periods mask，只在上涨段交易
  ↓ 优化器目标函数现包含绝对收益评分：S_performance = 0.6 × sharpe_score + 0.4 × return_score
  ↓ 输出: params.yaml
Step 4: 验证（Validation）
  ↓ 6 层验证: Regime CV → OOS → Walk-Forward → DSR → Monte Carlo → Industrial
  ↓ 输出: validation.yaml
Step 5: 归因（Attribution）
  ↓ 5 层归因: Signal → Horizon → Regime → Baseline → Operational
  ↓ 输出: attribution.md
Step 6: AlphaForge 报告（必须）
  ↓ 生成 train.html + oos.html
  ↓ 使用 Industrial 模式
Step 7: 准入评估
  ↓ 对照准入标准判定 pass/fail
  ↓ 通过 → 等待 Portfolio 构建
```

### 3.2 AlphaForge 报告要求（Step 6 详细）

**每个通过优化的策略都必须生成 AlphaForge 报告。**

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
    output_path="research/{path}/v{N}_{return}%/train.html",
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

---

## 指标面板（Indicator Panels）

每个策略应实现 `get_indicator_panels(datetimes)` 方法，用于在 AlphaForge HTML 报告中渲染指标可视化。

### 分类规则

- **Overlay（主图叠加）**：价格级指标 — EMA, SuperTrend, Bollinger, Donchian, KAMA, HMA, MAMA, POC
- **Subplot（独立副图）**：振荡器和量价指标 — RSI, MACD, ADX, Force Index, CMF, OBV, MFI, Chaikin 等
- **Signal**：策略信号 — 由 `backtest_runner` 自动追加为最后一个 subplot

### 实现示例

```python
def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
    return {
        "overlays": [
            self._make_overlay("EMA(20)", datetimes, self._ema, color="#ffab40"),
        ],
        "subplots": [
            self._make_subplot(
                "RSI(14)",
                [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc")],
                horizontal_lines=[30, 70],
                y_range=[0, 100],
            ),
        ],
    }
```

### 支持的样式

| 样式 | 用于 | 示例 |
|------|------|------|
| `line` | 普通折线 | EMA, RSI, MACD Line |
| `step` | 阶梯线 | SuperTrend, POC |
| `dash` | 虚线 | Bollinger 上下轨 |
| `bar` | 柱状图 | MACD Histogram（支持 color_positive/color_negative） |
| `area` | 填充面积 | Signal 强度 |

---

## 4. 策略设计要求

### 4.1 基类接口（必须遵守）

每个策略必须实现 `get_indicator_panels(datetimes)` 方法，提供指标面板可视化数据。

```python
from strategies.templates.base_strategy import QBaseStrategy

class MyStrategy(QBaseStrategy):
    # === 必填类属性 ===
    name: ClassVar[str] = "strong_trend_long_iron_daily_v1"
    regime: ClassVar[str] = "trending"          # "trending" | "mean_reversion"
    horizon: ClassVar[str] = "medium"           # "fast" | "medium" | "slow" | None (MR)
    signal_dimensions: ClassVar[list[str]] = ["momentum", "volume"]
    warmup: ClassVar[int] = 75

    # === 可优化参数（≤ 5 个，含 chandelier_mult）===
    fast_period: int = 12
    slow_period: int = 26
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        # 在此预计算所有指标数组
        self._macd_line, _, _ = macd(self._closes, self.fast_period, self.slow_period, 9)

    def _generate_signal(self, bar_index: int) -> float:
        # 返回 [-1, +1] 信号，warmup 期间返回 0
        if bar_index < self.warmup:
            return 0.0
        # ... 信号逻辑
        return signal

    def get_indicator_config(self) -> list[dict]:
        return [{"name": "MACD", "params": {"fast": self.fast_period, "slow": self.slow_period}}]
```

### 4.2 参数约束

- 每策略可优化参数 **≤ 5 个**（含 `chandelier_mult`）
- 参数范围窄（2-3x 默认值）
- 风控参数和信号参数一起优化

### 4.3 信号维度多样性

每个 timeframe 目录下的 ~20 个策略应覆盖不同信号维度：

| 类型 | 建议数量 | 信号维度 | 典型指标 |
|------|---------|---------|---------|
| 纯动量 | 4-5 | momentum | TSMOM, EMA Cross, SuperTrend, ADX |
| 动量 + 量价 | 5-6 | momentum + volume/OI | MACD+CMF, SuperTrend+OBV, EMA+OI flow |
| 动量 + 技术 | 4-5 | momentum + technical | MACD+RSI, Aroon+Force, ADX+Bollinger |
| 动量 + Carry | 3-4 | momentum + carry | TSMOM+Basis, EMA+Term Structure |
| 多维混合 | 2-3 | 3+ dimensions | MACD+CMF+RSI, SuperTrend+OI+Carry |

### 4.4 方向约束

- **long/ 目录下的策略：** 信号范围 `[0, 1]`，只做多
- **short/ 目录下的策略：** 信号范围 `[-1, 0]`，只做空
- **mean_reversion/ 目录下：** 信号范围 `[-1, 1]`，双向交易

---

## 5. 策略准入标准

### 5.1 硬筛条件（全部满足才能入选 Portfolio）

| 条件 | 阈值 | 检查来源 |
|------|------|---------|
| OOS Sharpe (industrial) | ≥ 0.5 | `validation.yaml` |
| Industrial Sharpe | ≥ 0.5 | `validation.yaml → industrial.industrial_sharpe` |
| Industrial 衰减 | ≤ 50% | `validation.yaml → industrial.decay_pct` |
| Max Drawdown | ≥ -25% | `validation.yaml → max_drawdown` |
| DSR | ≥ 0.95 | `validation.yaml → deflated_sharpe` |
| Bootstrap CI | 不跨零（非 FRAGILE） | `validation.yaml → bootstrap.verdict` |
| Regime CV | 非 FAIL | `validation.yaml → regime_cv.verdict` |
| 独立 Alpha | > 0 | `attribution.md → Baseline Decomposition` |
| 活跃度 | abs(return) > 0.1% | `oos.html → result metrics` |
| 2x 成本生存 | Sharpe@2x > 0 | 2 倍交易成本下仍盈利 |
| OOS 年化收益 | > 0% | `validation.yaml` → `oos_full_span.annualized_return` |
| Profit Factor | ≥ 1.3 | `validation.yaml` → `oos_full_span.profit_factor` |
| 单笔最大贡献 | < 30% 总收益 | `validation.yaml` → `max_single_trade_contribution` |

**WARNING 规则：** `Sharpe > 1.5 但年化收益 < 3% → WARNING`（高 Sharpe 但绝对收益过低，可能是低活跃度假象）

### 5.2 交易次数信心分级

交易次数不作为硬筛，而是影响权重上限：

| 信心等级 | Daily 交易次数 | 1H 交易次数 | 权重上限 |
|---------|:------------:|:----------:|:-------:|
| HIGH | ≥ 30 | ≥ 100 | 25% |
| MODERATE | 10-29 | 30-99 | 25% |
| LOW | < 10 | < 30 | **15%** |

### 5.3 组合适配检查（Portfolio Fit）

| 检查 | 阈值 | 说明 |
|------|------|------|
| 与现有组合相关性 | < 0.40 | 超过则不入选 |
| 边际 Sharpe 贡献 | > 0 | SR_candidate > ρ × SR_portfolio |
| 两两相关性矩阵 | 标记 ≥ 0.40 的对 | 高相关对降权 |

### 5.4 Portfolio 入选标准

通过准入的策略还需满足更高标准才能进入 Portfolio：
- OOS Total Return > 5%
- OOS Sharpe ≥ 1.0
- 两段 OOS 都盈利，单段占比 < 70%
- 与现有组合 corr < 0.5
- 加入后 Return 提升且 MaxDD 恶化 ≤ 3%
- 最多 8 个策略（Daily ≤ 5, 1H ≤ 3）
- 同一指标组合不重复

**重要：** Correlation 使用策略 returns（OOS回测日收益率）计算，不使用 forecast 值。Daily策略forecast在1H grid上forward-fill导致forecast correlation失真。

详见 `portfolio/{regime}/{direction}/{instrument}/selection_criteria.yaml`

---

## 6. Research 目录产物清单

每个策略完成开发流程后，其 research 目录应包含：

```
research/{regime}/{direction}/{instrument}/{timeframe}/v{N}_{+/-}{return}%/
├── params.yaml              # 必须 — 优化参数 + opt_score + is_robust
├── validation.yaml          # 必须 — 6 层验证结果 + hard_reject + soft_flags
├── attribution.md           # 必须 — 5 层归因分析
├── train.html               # 必须 — AlphaForge Industrial IS 回测报告
├── oos.html                 # 必须 — AlphaForge Industrial OOS 回测报告
└── holdout.html             # Portfolio 开封后 — AlphaForge Holdout 报告
```

### 6.1 params.yaml 格式

```yaml
best_params:
  fast_period: 11
  slow_period: 78
  signal_period: 15
  cmf_period: 55
  chandelier_mult: 1.856
opt_score: 0.620
is_robust: true
optimizer:
  coarse_trials: 30
  fine_trials: 50
  seeds: [42, 123, 456]
```

### 6.2 validation.yaml 格式

```yaml
hard_reject: false
reject_reasons: []
soft_flags: []

oos_full_span:
  total_return: 0.182
  annualized_return: 0.085
  sharpe: 1.394
  n_trades: 6
  max_drawdown: -0.12
  profit_factor: 1.85
  max_single_trade_pct: 0.22

oos_period_breakdown:
  - period: "2019-03-01 ~ 2019-11-30"
    return: 0.12
    sharpe: 1.8
    n_trades: 3
  - period: "2021-06-01 ~ 2022-02-28"
    return: 0.06
    sharpe: 0.9
    n_trades: 3

period_concentration_warning: false   # true 当单段贡献 > 70% 总收益

industrial_oos_sharpe: 1.381
is_sharpe: 0.856
dsr: 0.97

regime_cv:
  verdict: PASS
  fold_sharpes: [0.8, 1.2, 0.6]

industrial:
  industrial_sharpe: 1.381
  decay_pct: 0.009

bootstrap:
  verdict: ROBUST
  ci_lower: 0.3
  ci_upper: 2.1

walk_forward:
  efficiency: 0.65
  win_rate: 0.70

permutation:
  p_value: 0.02
```

### 6.3 attribution.md 格式

```markdown
# Attribution: v{N}

## Horizon Fingerprint
- Fast: 11.2% | Medium: 88.8% | Slow: 0.0%

## Baseline Decomposition
- TSMOM Beta: 59.2%
- Carry Beta: 0.0%
- Independent Alpha: 40.8%

## Industrial Decay
- Decay: -0.3% (positive = costs add alpha)

## Indicator Contributions
- MACD Line: 65% of signal variance
- CMF: 35% of signal variance
```

### 6.4 summary.yaml 格式（每个 timeframe 目录一个）

```yaml
- version: v1
  oos_sharpe: 1.394
  industrial_oos_sharpe: 1.381
  n_trades_oos: 6
  hard_reject: false
  robust: true
  score: 0.620
  independent_alpha: 0.036
  status: PASSED

- version: v2
  oos_sharpe: -1.051
  hard_reject: true
  status: FAILED
  reject_reason: negative OOS Sharpe
```

---

## 7. Portfolio 报告结构

Portfolio 级报告存放在 `reports/{regime}/{direction}/{instrument}/`：

| 文件 | 内容 |
|------|------|
| `portfolio_summary.html` | 主报告：合成权益曲线、相关性矩阵、权重饼图、vs TSMOM 基准 |
| `strategy_comparison.html` | 所有入选策略的横向对比表 |
| `coverage_matrix.html` | Regime 覆盖矩阵（检查是否有盲区） |
| `weights.yaml` | 最终权重分配（含信心等级、权重上限） |
| `validation_summary.yaml` | Portfolio 级验证结果（LOO, Bootstrap, 5D15指标评分） |

### 7.1 weights.yaml 格式

```yaml
method: equal_weight  # equal_weight | inverse_volatility | hrp_alpha_consistency
total_strategies: 8
timeframe_allocation:
  daily: 0.55
  1h: 0.45

strategies:
  - name: strong_trend_long_iron_daily_v1
    weight: 0.12
    confidence: HIGH
    max_weight: 0.25
    oos_sharpe: 1.394
    independent_alpha: 0.036

  - name: strong_trend_long_iron_1h_v33
    weight: 0.08
    confidence: MODERATE
    max_weight: 0.25
    oos_sharpe: 1.469
    independent_alpha: 0.15
```

---

## 8. Regime 与策略激活规则

| 基本面预判 | 激活策略集 | 资金比例 |
|-----------|----------|---------|
| strong_trend | 对应 regime 的 Trending 策略 | 100% |
| mild_trend | 对应 regime 的 Trending 策略 | 100% |
| mean_reversion | MR 策略集 | 100% |
| crisis | 所有策略集 | **50%**（强制减仓） |

**不做 Regime 间混合分配。** 基本面团队给确定性预判。

---

## 9. Timeframe 杠铃结构

基于 CTA 行业研究（CFA Institute 2026, Barbell Paper 2025），推荐：

| Timeframe | 风险预算 | 理由 |
|-----------|---------|------|
| **Daily** | 55% | 成本最低，容量最大，慢趋势 backbone |
| **1H** | 45% | 快速反应，与 daily 相关性最低（ρ≈0.3） |

4H/2H 作为可选扩展，但核心组合优先 Daily + 1H。

**日历时间对照：**

| Timeframe | Medium Horizon (60-125 bar) | 实际捕捉趋势 |
|-----------|---------------------------|------------|
| Daily | 60-125 天 | 3-6 个月 |
| 1H | 60-125 小时 | 2-4 周 |

---

## 10. 现有策略清单

> **当前状态：** Daily 30 策略，1H 30 策略（v1-v20 新开发 + v31-v50 旧版迁移）。
> **所有策略均已实现 `get_indicator_panels()` 方法**，支持 AlphaForge HTML 报告中的指标面板可视化。

### 10.1 Iron Ore / Long / Strong Trend

#### Daily（v1-v30）

| 版本 | OOS Sharpe | 状态 | 信号维度 |
|------|-----------|------|---------|
| v1 | 1.394 | PASSED | momentum + volume (MACD + CMF) |
| v5 | 0.440 | PASSED | momentum + technical (Aroon + Force) |
| v6 | 0.903 | PASSED | momentum + volume |
| v9 | 1.146 | PASSED | momentum + volume |
| v13 | 0.921 | PASSED | momentum + volume |
| v17 | 0.778 | PASSED | momentum + volume |
| v18 | 0.169 | PASSED | momentum + volume |
| v20 | 1.665 | PASSED | momentum + volume |
| v22 | 1.570 | PASSED | momentum + technical |
| v23 | 1.633 | PASSED | momentum + volume (SuperTrend + OI) |
| v27 | 1.104 | PASSED | momentum + volume |
| v28 | 0.808 | PASSED | momentum + volume |
| v2 | -1.051 | FAILED | — |
| v7 | 0.689 | FAILED | Industrial decay > 50% |
| v8 | -0.969 | FAILED | — |
| v11 | -1.982 | FAILED | — |
| v12 | -1.335 | FAILED | — |
| v14 | 1.578 | FAILED | Industrial decay > 50% |
| v21 | -0.333 | FAILED | — |
| v24 | 0.623 | FAILED | Industrial decay > 50% |
| v26 | -1.803 | FAILED | — |

#### 1H（v1-v20 新开发 + v31-v50 旧版，共 30 策略）

参见 `research/strong_trend/long/iron/1h/summary.yaml`

#### 2H（v51-v70）— 从 summary_2h.yaml

参见 `research/strong_trend/long/iron/2h/summary.yaml`

#### 4H（v71-v90）— 从 summary_4h.yaml

参见 `research/strong_trend/long/iron/4h/summary.yaml`

---

## 11. 快速参考

### 新建一个策略的 Checklist

- [ ] 确定 regime / direction / instrument / timeframe
- [ ] 在对应目录创建 `v{N}.py`
- [ ] 继承 `QBaseStrategy`，填写所有必填属性
- [ ] 实现 `on_init_arrays` + `_generate_signal` + `get_indicator_config`
- [ ] 实现 `get_indicator_panels()`
- [ ] 运行优化 → 保存 `params.yaml`
- [ ] 运行验证 → 保存 `validation.yaml`
- [ ] 运行归因 → 保存 `attribution.md`
- [ ] 生成 AlphaForge train.html + oos.html（**Industrial 模式，必须传 bar_data**）
- [ ] 对照准入标准判定 pass/fail
- [ ] 更新 summary.yaml

### 关键路径

```
v{N}.py → optimize → params.yaml → validate → validation.yaml
                                             → attribute → attribution.md
                                             → AlphaForge report → train.html + oos.html
                                             → 准入判定 → summary.yaml 更新
```

### 方向约束（目录路径自动推断）

- **long/ 目录下的策略：** `direction = "long"`，信号自动裁剪到 `[0, 1]`，只做多
- **short/ 目录下的策略：** `direction = "short"`，信号自动裁剪到 `[-1, 0]`，只做空
- **mean_reversion/ 目录下：** `direction = "both"`，信号范围 `[-1, 1]`，双向交易
- 基类 `QBaseStrategy` 在 `generate_signals()` 中自动执行方向裁剪，无需策略内部处理

### 品种命名

策略中使用 **ticker 代码**（如 `I` 而非 `iron`）作为品种标识。目录名使用小写 ticker。

### Timeframe 迁移规则

- 策略初始开发在某个 timeframe 下（如 `daily/v5.py`）
- 优化后如果发现该策略在其他 freq 上表现更好，**允许迁移**
- 迁移步骤：
  1. 将 `v{N}.py` 移动到目标 timeframe 目录
  2. 将 `research/.../v{N}_{return}%/` 整个目录移动到对应位置
  3. 更新策略的 `name` 属性（如 `..._daily_v5` → `..._4h_v5`）
  4. 更新目标目录的 `summary.yaml`

## 12. Position Sizing（Carver 连续调仓方式）

### Forecast Scale

策略层面仍然输出 [-1, +1] 原始信号。Signal Blender 通过 `forecast_scalar` 将每个策略的信号缩放到 Carver 标准 forecast 范围 [-20, +20]（10 = 平均信心，20 = 最大信心）。缩放公式：`forecast = signal × forecast_scalar`，其中 `forecast_scalar` 基于历史 `avg|signal|` 校准使 `avg|forecast| = 10`。

### 单策略层面

每根 bar 重新计算目标仓位，偏差 > 10% 时调仓：

```
target_lots = (forecast / 10) × (capital × TARGET_VOL) / (price × multiplier × ann_vol)
```

其中 `forecast` 为缩放后的信号（[-20, +20]）。单策略独立运行时 `forecast = signal × forecast_scalar`；Portfolio 层面由 Signal Blender 输出合并 forecast。

| 参数 | 值 | 说明 |
|------|-----|------|
| TARGET_VOL | 15% | 目标年化波动率 |
| VOL_LOOKBACK | 20 bar | 已实现波动率估计窗口 |
| MAX_MARGIN_UTIL | 80% | 保证金上限 |
| SIGNAL_THRESHOLD | 0.05 | 死区：|signal| ≤ 0.05 → 平仓（原始信号尺度） |
| REBALANCE_BUFFER | 10% | 仓位偏差 > 10% 才调仓（buffer zone / position inertia） |

### 调仓逻辑

| 情况 | 动作 |
|------|------|
| 无仓 + 信号 > 0.05 | 开仓 |
| 持仓 + 信号进入死区 | 立即平仓 |
| 持仓 + 方向翻转 | 平仓 + 反向开仓 |
| 持仓 + 同向偏差 > 10% | 加仓或减仓到目标 |
| 持仓 + 同向偏差 ≤ 10% | 不动（position inertia） |

### Portfolio 层面

使用 Signal Blender（Carver 标准 Forecast Combination Pipeline）：各策略信号 → Forecast Scaling (avg|f|=10) → Capping (±20) → 加权合并 → FDM → Re-cap → Direction Filter → 一个合并 forecast [-20, +20] → vol-target sizing → 一个净仓位。不按策略分配资金。多频率在 1H 网格上合并（daily 信号 forward-fill）。

---

### 禁止事项

- 不可因 OOS 结果修改参数（数据分割在 Phase 2 标注时完成，终身不变）
- 不可删除 `research_log/trials/` 中的试验记录（DSR 计算依赖完整记录）
- 不可提前开封 Holdout（仅在 Portfolio 最终确认时使用）
- Industrial 衰减 > 50% 的策略不入 Portfolio
