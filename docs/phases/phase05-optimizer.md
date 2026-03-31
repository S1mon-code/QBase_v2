# Phase 5: 优化器

**目标：** 在标注的 regime 时段上优化策略参数，5 维复合目标函数，两阶段 Optuna。

**依赖：** Phase 4（策略代码可运行）

---

## 核心设计：按 Regime 时段训练

```python
# optimizer/regime_optimizer.py
class RegimeOptimizer:
    """
    1. 读取 regime_labels/{symbol}.yaml
    2. 筛选 regime=X, direction=Y, split=train 的所有时段
    3. 含 buffer 时段
    4. 在这些时段上跑 Optuna
    """
```

---

## 5 维复合目标函数

```python
score = (0.40 * S_performance
       + 0.15 * S_significance
       + 0.15 * S_consistency
       + 0.15 * S_risk
       + 0.15 * S_alpha)
```

### 维度 1: S_performance (40%) — 风险调整收益

```python
# 粗调: tanh 压缩（防追极端 Sharpe）
S_performance_coarse = 10 * tanh(0.7 * sharpe)

# 精调: linear（追求绝对精度）
S_performance_fine = min(10, sharpe * 10 / 3)
```

### 维度 2: S_significance (15%) — 统计显著性

```python
# Lo (2002) 修正 Sharpe t-stat
t_stat = sharpe * sqrt(n_trades) / sqrt(1 + 0.5*skew*sharpe - (kurt-3)/4*sharpe**2)
S_significance = min(10, t_stat * 10 / 3)    # t > 3: 满分
```

### 维度 3: S_consistency (15%) — 时间一致性

```python
# 回测分 N 个等长窗口，计算每个窗口 Sharpe
win_rate = fraction(window_sharpe > 0)
cv = std(window_sharpes) / mean(window_sharpes)
consistency = win_rate * max(0, 1 - cv)
S_consistency = 10 * consistency
```

### 维度 4: S_risk (15%) — 尾部风险

```python
maxdd_score = max(0, 10 * (1 - abs(maxdd) / 0.40))    # 40% DD = 0 分
cvar_score = max(0, 10 * (1 + cvar_95 / 0.03))         # CVaR -3% = 0 分
S_risk = 0.6 * maxdd_score + 0.4 * cvar_score
```

### 维度 5: S_alpha (15%) — 超越 Baseline 增量

```python
alpha_sharpe = strategy_sharpe - tsmom_baseline_sharpe
S_alpha = max(0, min(10, alpha_sharpe * 10 / 1.0))    # alpha > 1.0: 满分
```

### 硬过滤

| 条件 | 动作 |
|------|------|
| 交易次数 < 频率门槛 (daily≥10, 4h≥20, 1h≥30) | 返回 -10 |
| S_alpha ≤ 0 | 返回 -5 |

---

## 两阶段优化

```
Phase 1: 粗调 (Coarse)
  30 trials, 全范围 TPE
  S_performance 用 tanh
  5 probe trials 早停
  死区 → 跳过精调

Phase 2: 精调 (Fine)
  50 trials, 围绕粗调最优 ±15%
  S_performance 用 linear
  1h+ 策略必须 Industrial 模式

Phase 3: 稳健性验证
  ±15% 邻域采样, max(20, n_params*5) 个样本
  ≥60% 邻居 > 最优50% → PLATEAU
  否则 → SPIKE
```

---

## 参数自动发现

从策略类的 type annotations + 默认值自动推导搜索空间：

```python
class TrendV1(BaseStrategy):
    st_period: int = 10        # → range [4, 30]
    st_mult: float = 3.0       # → 已知: [1.5, 5.0]
    chandelier_mult: float = 2.5  # → 已知: [1.5, 4.0]
```

已知参数映射（精确范围）见 optimizer/config.py。

---

## 多种子验证（可选）

3 seeds (42, 123, 456) 独立跑完整两阶段，取中位数。`--multi-seed` 开启。仅对 top 候选使用。

---

## 试验记录系统

**每次 trial 自动写入，不可删除：**

```yaml
# research_log/trials/trial_registry.yaml
total_trials: 0
trials:
  - id: "trial_0001"
    strategy: "trend_medium_v1"
    timestamp: "2026-04-01T10:30:00"
    params: {st_period: 10, ...}
    regime: "strong_trend"
    direction: "up"
    symbol: "RB"
    freq: "1h"
    sharpe: 1.45
    score: 7.2
    n_trades: 87
    status: "active"
```

---

## 代码结构

```
optimizer/
├── core.py               # 5 维目标函数 + 硬过滤
├── two_phase.py           # 粗调 → 精调 流程
├── robustness.py          # 高原检测 + 多种子
├── regime_optimizer.py    # 按 regime 时段筛选训练数据
├── param_discovery.py     # 参数自动发现
├── baseline.py            # TSMOM Baseline 计算（供 S_alpha）
├── trial_registry.py      # 试验记录（自动写入/读取）
└── config.py              # 已知参数范围 + 优化配置
```

---

## 风险

中。重写量大（v1 optimizer_core 1200 行），但逻辑清晰。新增的 S_significance 和 S_alpha 维度需要调试权重。
