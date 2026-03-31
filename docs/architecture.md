# QBase_v2 Architecture

## 系统定位

黑色系中国期货单品种多策略交易系统。核心思路：**基本面预判市场 regime → 匹配历史同类时段训练策略 → 部署对应策略集。**

不做跨品种套利。单品种内多策略信号混合，多周期同步运行。

---

## 数据流

```
基本面团队                    历史数据 (AlphaForge V6.0)
    │                              │
    ▼                              ▼
Regime 预判               Regime 历史标注
(mild_trend, up)          (Bry-Boschan + 人工)
    │                              │
    └──────────┬───────────────────┘
               │
    ┌──────────▼──────────┐
    │  Historical Matching │  匹配同类 regime 时段 (±2月buffer)
    │  → 训练数据选择       │  → train/oos/holdout 分割
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Strategy Pool      │  按 regime × horizon 组织
    │   4 信号维度          │  Momentum / Carry / Volume / Technical
    │   3 trend horizons   │  Fast / Medium / Slow
    │   4 周期同步          │  1h / 2h / 4h / daily
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Signal Blender     │  多策略信号加权合并 → 单一净信号
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Risk Pipeline      │
    │   Direction Filter   │  看多→max(0), 看空→min(0)
    │   Vol Targeting      │  target_vol / realized_vol
    │   Position Sizing    │  2% 单笔风险
    │   Chandelier Exit    │  regime-adaptive ATR 倍数
    │   Portfolio Stops    │  -10%/-15%/-20%/-5% 四级
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Execution          │  AlphaForge V6.0 Industrial
    └─────────────────────┘
```

---

## 模块依赖图

```
indicators/ ─────────────────────────────────────┐
                                                  │
config/ ──────┬──────────────────────────────────┐│
              │                                   ││
regime/ ──────┼── regime/labeler (Phase 2)        ││
              │   regime/matcher                  ││
              │   regime/schema                   ││
              │                                   ││
risk/ ────────┼── risk/chandelier (Phase 3)       ││
              │   risk/vol_targeting              ││
              │   risk/position_sizer             ││
              │   risk/directional_filter         ││
              │                                   ││
strategies/ ──┼── templates/ (Phase 4)          ◄─┘│
              │   baselines/                    ◄──┘
              │   trending/{fast,medium,slow}/
              │   mean_reversion/
              │
optimizer/ ───┼── core, two_phase (Phase 5)
              │   regime_optimizer
              │   trial_registry
              │
validation/ ──┼── regime_cv, oos (Phase 6)
              │   walk_forward, deflated_sharpe
              │   monte_carlo, permutation_test
              │
attribution/ ─┼── signal, horizon (Phase 7)
              │   regime, baseline
              │   operational, coverage
              │
portfolio/ ───┼── signal_blender (Phase 8)
              │   weights, scorer
              │   regime_allocator
              │
monitoring/ ──┼── decay_detector (Phase 11)
              │   regime_alert
              │   retirement
              │
pipeline/ ────── runner, cli (Phase 10)
```

---

## 核心设计决策

### 1. 基本面驱动 vs 技术面检测

**决策：** Regime 由基本面团队预判，不由技术面实时检测。

**理由：** 基本面团队对铁矿/螺纹的供需有专业判断力。技术面 regime 检测有滞后性（检测到趋势时趋势可能已经走了大半）。基本面可以提前预判。

**未来升级路径：** 基本面量化模型开发完成后，替换手动 YAML 输入。

### 2. Signal Blending vs 策略独立运行

**决策：** 同 regime 内策略信号先混合再交易（Signal Blending），不是各自独立下单。

**理由：** 减少对冲损耗（策略 A 做多 3 手 + 策略 B 做空 2 手 → 净做多 1 手，而不是 5 手总敞口）。降低交易成本。

### 3. 风控内置 vs 事后叠加

**决策：** Chandelier Exit 和 Vol Targeting 内置于策略模板，参数和信号参数一起优化。

**理由：** 如果不带风控优化出的参数，加风控后可能完全失效。v1 经验：宽止损(ATR>4.0)是趋势策略最重要的单一因素。

### 4. 多周期同步 vs 逐步扩展

**决策：** 每个策略从第一天就在 1h/2h/4h/daily 上同步运行。

**理由：** 跨周期一致性是策略稳健性的重要指标。一个只在 1h 上有效的策略可能过拟合到高频噪音。同步开发能更早发现这类问题。

### 5. TSMOM Baseline 作为零假设

**决策：** 每个策略必须 beat TSMOM Baseline。

**理由：** AQR 研究显示 TSMOM 解释了 CTA 行业 53-64% 的收益。如果策略不如 TSMOM，它没有增加独立信息。防止开发一堆"换名字的动量策略"。

### 6. Deflated Sharpe + 全试验记录

**决策：** 记录每一次优化试验，用 Deflated Sharpe Ratio 校正选择偏差。

**理由：** López de Prado (AQR) 的标准方法。测试 200 个策略选最好的 10 个，如果不校正，最好的那个可能只是噪音。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python |
| 回测引擎 | AlphaForge V6.0 (1505 tests, 95 品种, Industrial 模式) |
| 指标计算 | numpy + numba @njit |
| 优化 | Optuna TPE |
| 协方差估计 | Ledoit-Wolf (sklearn) |
| Portfolio 权重 | HRP (自实现) |
| 数据格式 | Parquet (AlphaForge), YAML (regime labels, config) |
| 报告 | Plotly HTML |

---

## 约束与限制

- **单品种只做一个方向** — 基本面 view 决定，不会同时做多做空
- **不做跨品种套利** — 但可用同板块品种辅助 Regime CV 训练
- **不做日内交易** — 最短周期 1h
- **切换频率周/月级** — 不做日级 regime 切换
- **策略参数 ≤ 5 个** — 防过拟合
- **Portfolio 上线标准 ≥ B+ (75分)** — 5 维 15 指标评分
