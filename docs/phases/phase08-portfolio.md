# Phase 8: Portfolio 构建

**目标：** Signal Blender + Regime Allocation 两层架构，策略权重分配 + 评分 + 止损。

**依赖：** Phase 7（归因完成的策略）

---

## 两层架构

```
Layer 1: Signal Blender（同品种同 Regime 内）
  多策略信号加权合并 → 单一净信号 → 单一净头寸

Layer 2: Regime Allocation（跨 Regime）
  基本面团队判断 → 激活对应 Regime 策略集 → 100% 资金
```

---

## Layer 1: Signal Blender（Carver 标准 Forecast Combination Pipeline）

策略输出原始信号 [-1, +1]，Signal Blender 通过 Carver/Man AHL 标准 forecast combination pipeline 合并为单一 forecast [-20, +20]。

### Forecast Combination Pipeline（6 步）

```
Step 1: Forecast Scaling  — signal × forecast_scalar → avg|forecast| = 10
Step 2: Forecast Capping  — clip 到 ±20
Step 3: Weighted Combination — 加权合并（NaN 策略自动剔除，权重重新归一化）
Step 4: FDM               — combined × FDM，补偿分散化损失
Step 5: Coverage Scaling  — 有信号策略 < 25% 时按覆盖率缩放
Step 6: Re-cap            — 再次 clip 到 ±20
```

**FDM (Forecast Diversification Multiplier):**
```
FDM = 1 / sqrt(w' × C × w)    # w = 权重向量，C = forecast 相关性矩阵
FDM = min(FDM, 2.5)           # 上限 2.5
```

### 多频率处理

Signal Blender 在 **1H 频率**上运行：
- 1H 策略：直接使用当前 bar 信号
- Daily 策略：信号 forward-fill 到 1H 网格
- 合并后的 forecast 在 1H 粒度上更新

### 3 阶段渐进权重

| Stage | 方法 | 适用 |
|:-----:|------|------|
| 1 | Equal Weight | 策略 < 5 个（初期） |
| 2 | Inverse Volatility | 策略 5-10 个 |
| 3 | HRP × Alpha × Consistency | 成熟期 |

### Stage 3 权重公式

```python
w_hrp = hrp_weights(ledoit_wolf_cov(returns))
alpha_factor = {v: max(0.2, indep_alpha[v] / max_alpha) for v in strategies}
consistency_factor = {v: wf_win_rate[v] for v in strategies}

w = {v: w_hrp[v] * alpha_factor[v] * consistency_factor[v] for v in strategies}
w = normalize(w)
w = clip_and_redistribute(w, max_weight=0.25)
```

### Horizon 分散约束

```python
# 每个 Horizon 至少 15% 权重
# 不足 → 从最大 Horizon 转移
MIN_HORIZON_WEIGHT = 0.15
```

### Forecast → Position Sizing

```python
forecast = pipeline_output                                    # [-20, +20]
forecast = directional_filter(forecast, fundamental_view)
lots = (forecast / 10) * (capital * target_vol) / (price * multiplier * ann_vol)
lots = buffer_zone(lots, current_lots, buffer=0.10)           # 10% position inertia
lots = clip(lots, max_position_by_margin)
```

---

## Layer 2: Regime Allocation

```
基本面预判 = long → 激活 Long 策略集, 100% 资金
基本面预判 = short → 激活 Short 策略集, 100% 资金
```

**不做 Regime 间混合分配。** 基本面团队是确定性预判，不是概率分配。未来接基本面量化模型输出概率时可升级。

---

## 策略筛选（进入 Portfolio 前）

| 条件 | 类型 |
|------|------|
| Regime CV = PASS | 硬筛 |
| Industrial Sharpe > 0 | 硬筛 |
| DSR > 0.95 | 硬筛 |
| Bootstrap CI 不跨零 | 硬筛 |
| 活跃度 abs(return) > 0.1% | 硬筛 |
| 独立 Alpha > 0（归因 Layer D） | 硬筛 |

---

## Portfolio 验证

1. **Leave-one-out** — 去掉任一策略 Sharpe 不应暴跌
2. **Bootstrap CI** — 95% 下界 > 0
3. **选择稳定性** — 扰动 100 次，CORE 策略 > 50%
4. **Regime 覆盖矩阵** — 无 RED FLAG

---

## 5 维 15 指标评分

| 维度 | 权重 | 指标 |
|------|:----:|------|
| 收益风险比 | 35% | Sharpe, Calmar, MaxDD, 回撤持续天数, CVaR 95% |
| 信号质量 | 25% | 平均独立 Alpha, Horizon 分散度, vs TSMOM 增量 |
| 组合效率 | 20% | 平均相关性, 回撤重叠率, Portfolio/Best Single, 正 Sharpe 比例 |
| 稳健性 | 15% | Bootstrap CI 宽度, CORE 占比, Permutation p |
| 实操性 | 5% | 策略数量, 最大权重, Industrial 衰减 |

**通过标准：≥ B+ (75分)**

---

## 再平衡

- Signal Blender 权重：月度更新（与基本面 review 周期同步）
- 只在有新的验证数据或策略增减时重新计算权重
- 不做日级动态调整（避免过度交易）

---

## 策略退役

| 条件 | 动作 |
|------|------|
| 滚动 6 月 Sharpe < 0 | 降权 50%，标注"观察" |
| 连续 3 月亏损 | 降权 50% |
| 滚动 12 月 Sharpe < -0.5 | 移除 |
| MaxDD 超过回测 1.5 倍 | 立即移除 |
| 市场结构变化 | 重新回测，不通过则移除 |

退役策略保留代码，文件头标注 `# RETIRED: YYYY-MM-DD 原因`。

---

## 代码结构

```
portfolio/
├── signal_blender.py      # Signal Blending
├── weights.py             # EW / IV / HRP+Alpha+Consistency
├── regime_allocator.py    # Regime 激活/休眠
├── constraints.py         # Horizon 分散 + 权重上限
├── selection.py           # 策略筛选 (DSR + 活跃度 + Alpha)
├── validation.py          # LOO + Bootstrap + 稳定性
├── scorer.py              # 5 维 15 指标评分
├── stops.py               # 4 级组合止损
├── rebalance.py           # 再平衡逻辑
├── retirement.py          # 策略退役检测
└── report.py              # Portfolio 报告
```

---

## 当前实现结果

> **2026-04-01** — Portfolio 构建完成。
>
> **当前组合：** daily_v27 + 1h_v18
> - **Sharpe = 1.979**
> - **Return = +64.64%**
> - **MaxDD = 6.36%**
> - **Calmar = 4.314**
> - 策略间 return correlation = 0.099
>
> **Selection：** 使用 return correlation（threshold < 0.5），不使用 forecast correlation。
>
> **Signal Blending Report：** 使用 AlphaForge `generate_signal_blend_report()` 生成，包含：
> - `portfolio.html` — 主报告（净信号权益曲线 + 各策略叠加对比 + 信号分解 + 相关性矩阵）
> - 各策略独立报告链接（strategy_links 参数传入）
>
> **报告结构：**
> ```
> reports/
> ├── portfolio.html                    # Signal Blending 主报告
> └── long/I/
>     ├── daily/v27/oos.html           # 策略独立报告
>     └── 1h/v18/oos.html             # 策略独立报告
> ```

---

## 风险

中。Signal Blender 是新架构，需要验证合并信号后的表现是否优于策略独立运行。HRP 在策略数少时可能不稳定（先用 Equal Weight）。
