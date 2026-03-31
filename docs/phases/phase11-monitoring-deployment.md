# Phase 12: 监控 + 实盘部署

**目标：** Paper Trading → 小资金实盘 → 正式实盘的路径，持续监控 + Alpha 衰减检测。

**依赖：** Phase 8（Production-ready Portfolio）

---

## 部署路径

```
回测验证 → Paper Trading → 小资金实盘 (10-20%) → 正式实盘
```

| 阶段 | 环境 | 持续时间 | 晋升标准 | 淘汰条件 |
|------|------|---------|---------|---------|
| Paper Trading | AlphaForge `af paper` | ≥ 2 个月 | 表现与回测一致 | 偏差 > 30% |
| 小资金实盘 | 真实资金 10-20% | ≥ 3 个月 | Sharpe > 0 | 持续亏损 / 风控触发 |
| 正式实盘 | 全额资金 | 持续 | — | 退役标准触发 |

---

## Paper Trading

```bash
af paper strategy.py --symbols RB --freq 1h --data-dir live_data/
af paper-status    # 查看持仓/权益
af paper-stop      # 停止
```

**监控指标：**
- 实际 vs 回测 Sharpe 偏差
- 实际 vs 回测交易频率偏差
- 滑点实际 vs 假设
- 信号延迟

---

## 持续监控

### 实时指标

| 指标 | 计算 | 黄灯 | 红灯 |
|------|------|------|------|
| 滚动 Sharpe (60日) | 60 日滚动 | < 0 持续 10 天 | < 0 持续 20 天 |
| 实际 vs 回测偏差 | (实际 - 回测) / 回测σ | > 1.5σ | > 2σ |
| MaxDD vs Monte Carlo | 当前 DD vs MC 95th | 超过 MC 中位数 | 超过 MC 95th |
| 交易频率 | 实际 vs 历史均值 | 偏差 > 50% | 偏差 > 100% |

### Alpha 衰减检测

```python
# monitoring/decay_detector.py
# 滚动 IC (Information Coefficient) 检测
# IC 持续下降 → Alpha 正在衰减

rolling_ic = rolling_correlation(signal, forward_returns, window=60)
if trend(rolling_ic, lookback=90) < -threshold:
    alert("Alpha decay detected")
```

### Regime 转换警报

```python
# monitoring/regime_alert.py
# 实际市场行为 vs 基本面预判 regime 是否匹配

if actual_vol > crisis_threshold and current_regime != "crisis":
    alert("Market behavior inconsistent with assigned regime")
```

---

## 策略退役（自动触发）

| 条件 | 动作 |
|------|------|
| 滚动 6 月 Sharpe < 0 | 降权 50%，标注"观察" |
| 连续 3 月亏损 | 降权 50% |
| 滚动 12 月 Sharpe < -0.5 | 移除 |
| MaxDD 超过回测 1.5 倍 | 立即移除 |
| Alpha 衰减检测触发 | 降权 + 调查 |

---

## 代码结构

```
monitoring/
├── paper_monitor.py       # Paper Trading 偏差监控
├── live_monitor.py        # 实盘指标监控
├── decay_detector.py      # Alpha 衰减检测
├── regime_alert.py        # Regime 转换警报
├── retirement.py          # 策略退役自动检测
└── dashboard.py           # 监控仪表板
```

---

## 交付物

1. Paper Trading 监控工具
2. Alpha 衰减检测
3. Regime 转换警报
4. 策略退役自动检测
5. 监控仪表板

---

## 风险

低。AlphaForge 已有 Paper Trading 基础设施。监控逻辑相对简单。
