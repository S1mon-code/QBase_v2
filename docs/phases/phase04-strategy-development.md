# Phase 4: 策略模板 + 第一批策略

**目标：** 定义信号接口，建立 TSMOM Baseline，在 RB 全周期（1h/2h/4h/daily）上同步开发策略。

**依赖：** Phase 2（regime 标注 + 训练数据）, Phase 3（风控模块）

**关键变更：** 策略从第一天就在 1h/2h/4h/daily 全周期上同步开发和验证，不单独设"扩展周期"阶段。每个策略在多周期上的表现差异本身是重要的研究信息。

---

## 4 维信号体系

每个策略从以下 4 个维度中选 1-3 个：

| 信号维度 | Alpha 来源 | 指标 |
|---------|-----------|------|
| **Momentum** | 初始低反应 + 羊群效应 | TSMOM, SuperTrend, EMA, ADX |
| **Carry** | 持有成本 + 供需预期 | 期限结构, 基差率, Roll Yield |
| **Volume/OI** | 散户逆势交易（中国市场特有） | Volume Momentum, OI 变化, Force Index |
| **Technical** | 超买超卖回归 | RSI, BB, Keltner, CCI |

**规则：**
- Trending 策略必须包含 Momentum 维度
- Carry 可单独使用或与 Momentum 混合
- 总参数 ≤ 5 个（含 chandelier_mult）

---

## 信号接口规范（Signal Blender 兼容）

所有策略必须实现统一接口：

```python
class BaseStrategy(TimeSeriesStrategy):
    # 子类必须定义
    regime: str          # "trending" / "mean_reversion"
    horizon: str         # "fast" / "medium" / "slow" / None(MR)
    signal_dimensions: list  # ["momentum", "carry"] 等

    @abstractmethod
    def _generate_signal(self, context) -> float:
        """返回 -1.0 到 +1.0 的信号强度
        正 = 做多倾向, 负 = 做空倾向, 0 = 无信号
        方向 filter 由外层处理，策略只管信号。
        """

    # 模板自动处理:
    # - Chandelier Exit 检查
    # - 仓位计算
    # - 下单执行
```

---

## TSMOM Baselines（第一步，在任何策略之前建立）

```python
# strategies/baselines/tsmom_fast.py
class TSMOMFast(BaseStrategy):
    """1 月 TSMOM — Fast Horizon Baseline"""
    regime = "trending"
    horizon = "fast"
    lookback = 20  # ~1 个月交易日

    def _generate_signal(self, context):
        returns = context.get_close_array(self.lookback)
        cum_return = (returns[-1] / returns[0]) - 1
        return 1.0 if cum_return > 0 else -1.0

# strategies/baselines/tsmom_medium.py — 3 月 (60 bars)
# strategies/baselines/tsmom_slow.py — 12 月 (250 bars)
```

**每个新策略必须 beat 对应 Horizon 的 TSMOM Baseline 才有存在价值。**

---

## 单策略开发流程（7 步）

### Step 1: 经济逻辑假设

每个策略必须回答"谁在亏钱给我"。写不出 → 不开发。

### Step 2: 指标选择

从 4 维信号体系中选 1-3 个维度，组合有逻辑约束。

### Step 3: 写策略代码

继承 `BaseStrategy`，实现 `_generate_signal`。风控内置（chandelier_mult 是参数）。必须用 `on_init_arrays` 预计算。

### Step 4: 裸逻辑验证

默认参数，RB 全历史：

| 指标 | 最低要求 |
|------|---------|
| 交易次数 | ≥ 30（1h 频率） |
| Profit Factor | ≥ 1.05 |
| 权益曲线 | 大致向上 |

不通过 → 微调或丢弃。

### Step 5: vs TSMOM Baseline 对比

在 RB 全历史上对比对应 Horizon 的 TSMOM Baseline。如果不如 Baseline → 策略没有增量价值。

### Step 6: Regime 时段验证

只在标注的 train split regime 时段上跑。Sharpe 应比全历史更好。

### Step 7: 记录到 research_log + 判定

---

## Round 1: 第一批策略（RB, 全周期同步）

**3 个 Baseline + 8 个策略 × 4 个周期：**

| 版本 | 类型 | Horizon | 核心信号 |
|------|------|---------|---------|
| baseline_fast | TSMOM | Fast | 1月 return sign |
| baseline_medium | TSMOM | Medium | 3月 return sign |
| baseline_slow | TSMOM | Slow | 12月 return sign |
| v1 | Momentum+Carry | Medium | TSMOM 3M + 期限结构 |
| v2 | Momentum+Volume | Medium | TSMOM 3M + VolMom |
| v3 | Pure Carry | Medium | 期限结构信号 |
| v4 | Technical | Medium | SuperTrend + VolMom |
| v5 | Blended | Medium | TSMOM + Carry + VolMom 混合 |
| v6 | Momentum | Fast | 短期动量 + Volume Spike |
| v7 | Momentum | Slow | Donchian + Ichimoku |
| v8 | MR | — | BB + RSI 极值 |

**每个策略同时在 1h / 2h / 4h / daily 上运行和验证。**

### 多周期同步开发流程

```
每个策略:
  1. 代码写一份（策略逻辑不变）
  2. 同时在 4 个周期上裸逻辑验证
  3. 记录每个周期的 Sharpe / PF / trades
  4. 同时在 4 个周期上优化（参数可以不同）
  5. 跨周期一致性检查:
     - 相邻周期 Sharpe 应同方向（可以低，但不应翻负）
     - 如果 1h Sharpe=1.5 但 daily=-0.3 → 策略可能过拟合到高频噪音
  6. 不同周期可能有不同的最优参数 → 分别保存
```

### Industrial 衰减预期（参考）

| 周期 | 预期衰减 | 含义 |
|------|:-------:|------|
| 1h | 25-55% | 必须在 Industrial 下优化 |
| 2h | 15-35% | 必须在 Industrial 下优化 |
| 4h | 10-25% | 粗调 Basic, 精调 Industrial |
| daily | 5-10% | 可全程 Basic, 最终 Industrial 验证 |

**Round 1 的唯一目的是验证 pipeline 能跑通。** 不追求策略质量。

---

## 策略来源

1. **v1 经验迁移** — VolMom, SuperTrend 等逻辑（不复制代码）
2. **信号维度组合** — 有逻辑约束的组合，不暴力穷举
3. **Research** — 学术论文/业界实践中对黑色系有效的策略
4. **Carry 维度** — v1 没有，v2 新增的独立 alpha 来源

---

## 交付物

1. `strategies/templates/base_strategy.py` — 统一信号接口
2. `strategies/templates/trending_template.py` — 趋势策略模板
3. `strategies/templates/mean_reversion_template.py` — 均值回归模板
4. `strategies/baselines/tsmom_*.py` — 3 个 TSMOM Baseline
5. `strategies/trending/medium/v1-v5.py` — 第一批趋势策略
6. `strategies/trending/fast/v6.py` + `slow/v7.py`
7. `strategies/mean_reversion/v8.py`
8. 每个策略的 research_log 记录
9. 测试覆盖

---

## 风险

中。策略开发是核心工作。v1 经验显示约 50% 策略在裸逻辑阶段被淘汰。Carry 指标是否在黑色系有效需要验证。
