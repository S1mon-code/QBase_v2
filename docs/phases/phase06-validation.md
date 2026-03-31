# Phase 6: 验证体系

**目标：** 6 层验证确保策略不是过拟合产物。

**依赖：** Phase 5（优化完成的策略）

---

## 6 层验证 Pipeline

```
Layer 1: Regime CV ──────── FAIL → 淘汰
Layer 2: OOS 验证 ──────── 标注但不淘汰
Layer 3: Walk-Forward ───── 标注"参数不稳定"
Layer 4: Deflated Sharpe ── 标注"可能假阳性"
Layer 5: Monte Carlo ────── FRAGILE → 不入 Portfolio
Layer 6: Industrial + 压力 ─ 衰减>50% → 不入 Portfolio
```

---

### Layer 1: Regime CV（训练集内跨时段一致性）

```
输入: 同类 regime 的 train split 时段 (N 段)

LOO (N ≤ 10): 留 1 段测试, 其余训练, 轮转
K-Fold (N > 10): 随机分 K 组

通过: PASS (Mean Sharpe > 0.3 且 Win Rate ≥ 50%)
      MARGINAL (Mean Sharpe > 0 且 Win Rate ≥ 33%)
      FAIL → 不进入 Layer 2

单品种样本不够时: 允许同板块品种辅助训练
  RB 时段不够 → 加 HC 的同类 regime 时段
  但 OOS/Holdout 必须只用目标品种
```

### Layer 2: OOS 验证（锁定参数样本外表现）

```
用优化参数跑 OOS split 时段

记录:
  OOS Sharpe
  WF Ratio = OOS_Sharpe / IS_Sharpe (≥ 0.5 合格)
  行为一致性（交易频率、持仓时间 vs IS）
  Industrial 模式 OOS Sharpe

不淘汰负 Sharpe（Portfolio 可能需要对冲）
标注: "疑似过拟合" / "行为异常" / "测试集偏高"
```

### Layer 3: Walk-Forward（参数时间稳定性）

**三种模式：**

| 模式 | 做法 | 适用 |
|------|------|------|
| Rolling WF | 固定 5年IS → 1年OOS 滚动 | 默认 |
| Expanding WF | IS 不断扩大 → 1年OOS | 信任更多数据 |
| **Regime-Aware WF** | 按 regime 时段滚动（不按固定时间） | **QBase_v2 主推** |

```
Regime-Aware WF:
  [regime段1→2→3训练] → [regime段4测试]
  [regime段1→2→3→4训练] → [regime段5测试]

通过: Win Rate ≥ 50%, Mean OOS Sharpe > 0
```

### Layer 4: Deflated Sharpe Ratio（校正多重检验）

```python
# López de Prado (2014)
# 从 trial_registry 读取全部试验次数
DSR = Prob(SR* > 0 | correction for N trials, skew, kurtosis)

通过: DSR > 0.95 (< 5% 假阳性概率)
```

**前提：** 优化器的试验记录系统（Phase 5）必须完整。

### Layer 5: Monte Carlo（双重）

**5a: Bootstrap（结果稳定性）**
```
1000 次有放回重采样交易序列
95% CI 下界 > 0: ROBUST
CI 跨零: FRAGILE → 不入 Portfolio
```

**5b: Permutation Test（信号真实性）**
```
1000 次打乱价格 returns 序列 → 同策略跑
p-value = 比例(random_sharpe > real_sharpe)
p < 0.05: 信号有真 edge
p > 0.10: 标注"信号可能无效"
```

### Layer 6: Industrial + 压力测试

```
Industrial 衰减:
  < 10%: 正常
  10-30%: 可接受
  30-50%: 需 Industrial 下重新优化
  > 50%: 不入 Portfolio

压力测试:
  滑点 ×2: Sharpe 衰减 < 30% → LOW
  相邻周期: 1h 策略跑 2h/4h
  相似品种: RB → HC
  极端行情: 2020/2021/2022
  成本 ×2: 手续费翻倍
```

---

## 硬淘汰 vs 软标注

| 层 | 硬淘汰 | 软标注 |
|----|--------|--------|
| Regime CV | FAIL | MARGINAL |
| OOS | — | 负 Sharpe, WF Ratio < 0.5 |
| Walk-Forward | — | Win Rate < 50% |
| Deflated Sharpe | — | DSR < 0.95 |
| Bootstrap | CI 跨零 (FRAGILE) | — |
| Permutation | — | p > 0.10 |
| Industrial | 衰减 > 50% | 衰减 30-50% |

---

## 代码结构

```
validation/
├── regime_cv.py          # Layer 1
├── oos_validator.py       # Layer 2
├── walk_forward.py        # Layer 3 (Rolling/Expanding/Regime-Aware)
├── deflated_sharpe.py     # Layer 4
├── monte_carlo.py         # Layer 5a Bootstrap
├── permutation_test.py    # Layer 5b Permutation
├── industrial_check.py    # Layer 6a
├── stress_test.py         # Layer 6b
├── pipeline.py            # 6 层编排
└── report.py              # 验证报告
```

---

## 风险

中。Deflated Sharpe 和 Permutation Test 是新增模块，需要仔细实现和调试。Regime-Aware WF 是原创设计，需要验证其合理性。
