# Phase 3: 风控模块

**目标：** Chandelier Exit + Vol Targeting + 仓位管理 + 方向 filter，独立于策略但内置于策略模板。

**依赖：** Phase 1（配置 + 指标库）

**与 Phase 2 可并行开发。**

---

## 核心原则

**风控从策略开发第一天就内置，不是事后叠加。** 优化时风控参数和信号参数一起跑 Optuna。

---

## Chandelier Exit（regime-adaptive）

```python
# risk/chandelier.py
class ChandelierExit:
    """
    多头: stop = highest_since_entry - atr_mult * ATR(14)
    空头: stop = lowest_since_entry + atr_mult * ATR(14)
    """
```

**ATR 倍数由 Regime 决定：**

| Regime | 方式 | ATR 倍数范围 | 优化建议 |
|--------|------|:----------:|---------|
| Strong Trend | 从极值回撤 | 2.5-3.5 | Optuna 范围 [2.0, 4.0] |
| Mild Trend | 从极值回撤 | 2.0-2.5 | Optuna 范围 [1.5, 3.0] |
| Mean Reversion | 从入场价固定 | 1.5-2.0 | Optuna 范围 [1.0, 2.5] |
| Crisis | 收紧 + 时间止损 | 1.5 固定 | 不优化 |

**Crisis 时间止损：** 持仓超过 N bars 无盈利 → 强制平仓。N 可优化。

---

## Vol Targeting（连续波动率缩放）

```python
# risk/vol_targeting.py
class VolTargeting:
    """
    position_scale = target_vol / realized_vol

    realized_vol: 指数加权移动标准差, halflife=60天
    target_vol: 10-15% 年化（config/settings.yaml）

    波动率高 → 自动减仓
    波动率低 → 自动加仓
    """
```

**叠加极端风控层：**
```python
if atr_percentile > 90:       # 极端高波
    position_scale *= 0.5     # 额外减半
elif atr_percentile > 80:
    position_scale *= 0.75
```

---

## Position Sizing

```python
# risk/position_sizer.py
class PositionSizer:
    """
    lots = (equity * risk_pct) / (stop_distance * multiplier)
    lots = min(lots, max_lots_by_margin)
    lots = max(1, lots)

    risk_pct: 0.02 (2% 单笔风险)
    stop_distance: chandelier_exit 计算的止损距离
    max_lots_by_margin: equity * 0.30 / margin_per_lot
    """
```

**Net Position 原则：** 单品种同一时间只有一个方向。Signal Blender 输出净信号后统一计算仓位。

---

## Directional Filter

```python
# risk/directional_filter.py
class DirectionalFilter:
    """
    读取 config/fundamental_views.yaml

    LONG_ONLY:  max(0, raw_signal)
    SHORT_ONLY: min(0, raw_signal)
    NEUTRAL:    raw_signal (不约束)
    """
```

**应用位置：** Signal Blender 输出后、Vol Targeting 之前。

---

## Portfolio Stops

```python
# risk/portfolio_stops.py
class PortfolioStops:
    """
    4 级组合止损:
    预警:    回撤 -10% → 日志 + 检查
    减仓:    回撤 -15% → 所有仓位减半
    熔断:    回撤 -20% → 全平, 人工审查
    单日熔断: 日亏 -5%  → 当日全平, 次日恢复

    恢复规则:
    减仓后: 回撤收窄至 -10% 恢复正常
    熔断后: 人工确认后重启
    连续 2 次 -20%: 停止 Portfolio, 重新优化
    """
```

---

## 敞口限制汇总

| 限制 | 数值 |
|------|------|
| 单笔最大亏损 | 权益 2% |
| 单策略保证金 | 权益 30% |
| 总保证金 | 权益 80% |
| 单日最大亏损 | 权益 5% |
| 组合熔断 | 权益 20% |

---

## 交付物

1. `risk/chandelier.py` — Chandelier Exit（regime-adaptive, 支持优化）
2. `risk/vol_targeting.py` — 连续波动率缩放
3. `risk/position_sizer.py` — 仓位计算
4. `risk/directional_filter.py` — 基本面方向裁剪
5. `risk/vol_classifier.py` — 动态波动率分级（ATR percentile）
6. `risk/portfolio_stops.py` — 组合级止损
7. 测试覆盖

---

## 风险

低。逻辑清晰，无复杂算法。Chandelier Exit 和 Vol Targeting 都是成熟方法。
