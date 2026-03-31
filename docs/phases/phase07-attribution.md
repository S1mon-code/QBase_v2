# Phase 7: 归因分析

**目标：** 5 层归因理解策略 Alpha 来源，指导 Portfolio 决策。

**依赖：** Phase 6（验证通过的策略）

---

## 5 层归因

```
Layer A: Signal Attribution     → 每个信号维度贡献了多少
Layer B: Horizon Attribution    → Alpha 来自 Fast/Medium/Slow 哪个周期
Layer C: Regime Attribution     → 策略在哪些 regime 下赚钱/亏钱
Layer D: Baseline Decomposition → 收益中多少是 Beta, 多少是独立 Alpha
Layer E: Operational Attribution → 成本对收益的侵蚀
```

---

### Layer A: Signal Attribution（Shapley + Ablation）

**信号 ≤ 4 个 → Shapley Value（精确分配，无残差）**

```
Shapley(signal_i) = 加权平均 signal_i 在所有联盟中的边际贡献
Σ Shapley(signal_i) = 总 Sharpe（精确）
```

**信号 > 4 个 → Ablation（逐个关闭，看 Sharpe 下降）**

```
Contribution_i = Baseline_Sharpe - Ablated_Sharpe_i
```

**贡献度解读：**

| 占比 | 含义 | 行动 |
|:----:|------|------|
| > 60% | 核心依赖 | 评估单点失效风险 |
| 5-60% | 有效贡献 | 保留 |
| < 5% | 冗余 | 考虑移除，简化策略 |

### Layer B: Horizon Attribution（CTA 标准方法）

```python
strategy_returns = α + β_fast*TSMOM_1M + β_medium*TSMOM_3M + β_slow*TSMOM_12M + ε

# β 的相对大小 = Horizon 指纹
# ε = 独立 Alpha（无法被 TSMOM 解释）
# R² = TSMOM 解释了多少收益变化
```

**意义：**
- 两个策略 Horizon 指纹相同 → Portfolio 中冗余
- 独立 Alpha (ε) 很小 → 策略没有超越 TSMOM 的价值

### Layer C: Regime Attribution（用标注标签）

```python
for trade in trades:
    trade.regime = lookup_regime(trade.entry_time, regime_labels)

# 按 4 个 regime 聚合: n_trades, win_rate, avg_pnl, total_pnl
# 新增: buffer/unlabeled 分类 → 看策略在 regime 过渡期表现
```

### Layer D: Baseline Decomposition（最重要）

```python
strategy_return = α + β_tsmom*TSMOM_returns + β_carry*Carry_returns + ε

# 分解:
# β_tsmom * TSMOM = 动量 Beta 贡献
# β_carry * Carry = 期限结构 Beta 贡献
# α + ε = 独立 Alpha（策略真正的独特价值）
```

**独立 Alpha 直接影响 Portfolio 权重（Phase 8）。**

### Layer E: Operational Attribution

```
Industrial 衰减分解:
  Basic Sharpe → 滑点成本 → 价差成本 → 换仓成本 → 锁板/拒单 → Industrial Sharpe
```

---

## 归因决策矩阵

| 归因发现 | 行动 |
|---------|------|
| 某信号 Shapley < 5% | 移除信号，简化策略 |
| 独立 Alpha < 2% 年化 | Portfolio 降权 |
| Horizon 指纹与另一策略 > 0.9 相关 | Portfolio 只保留一个 |
| 某 regime 下胜率 < 30% | 标注 regime 盲区 |
| 所有策略同一 regime 亏损 | **RED FLAG** |

---

## 归因报告格式

```markdown
# {strategy} {symbol} 归因报告

## Signal Attribution
- dominant: {指标名} (XX.X%)
- redundant: {指标名} (X.X%)

## Horizon Fingerprint
- Fast: XX% | Medium: XX% | Slow: XX%
- Independent Alpha: XX%
- R²: X.XX

## Regime Attribution
- best: {regime} (XX% WR, +XX%)
- worst: {regime} (XX% WR, -XX%)

## Baseline Decomposition
- TSMOM Beta: XX% of return
- Carry Beta: XX% of return
- Independent Alpha: XX% of return

## Operational
- Industrial 衰减: XX%

## 建议
- Portfolio 权重建议
- 策略简化建议
```

---

## 代码结构

```
attribution/
├── signal.py            # Layer A: Shapley + Ablation
├── horizon.py           # Layer B: TSMOM horizon 回归
├── regime.py            # Layer C: regime 标签归因
├── baseline.py          # Layer D: TSMOM + Carry 分解
├── operational.py       # Layer E: Industrial 成本分解
├── coverage.py          # 覆盖矩阵 + RED FLAG
├── drawdown.py          # 回撤归因
├── decay.py             # Alpha 衰减（滚动 IC）
├── batch.py             # 批量归因
└── report.py            # 归因报告生成
```

---

## 风险

低。Shapley Value 对 ≤ 4 信号计算量可接受。Horizon/Baseline 回归是标准线性回归。
