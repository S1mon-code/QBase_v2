# QBase_v2 策略开发指引

> 快速参考卡。完整开发流程见 [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)，Portfolio 标准见 [PORTFOLIO.md](PORTFOLIO.md)。

---

## 1. 目录架构

### strategies/

```
strategies/
├── templates/                          # 基类模板
│   ├── base_strategy.py
│   ├── trending_template.py
│   └── mean_reversion_template.py
├── baselines/                          # TSMOM Baselines
├── mild_trend/                         # I 策略 (long 150 + short 40)
│   ├── long/I/{daily,1h,2h,4h}/
│   └── short/I/{daily,1h,2h,4h}/
├── strong_trend/                       # AG 策略 (long 40 + short 40)
│   ├── long/AG/{daily,1h,2h,4h}/
│   └── short/AG/{daily,1h,2h,4h}/
├── mean_reversion/                     # 无方向分层，双向交易
│   └── {instrument}/{daily,1h,2h,4h}/
└── crisis/
    ├── long/{instrument}/{timeframes}/
    └── short/{instrument}/{timeframes}/
```

**层级规则：** `regime / direction / instrument / timeframe / v{N}.py`

**Mean Reversion 例外：** `mean_reversion / instrument / timeframe / v{N}.py`（双向交易，无方向分层）

### research/（镜像 strategies/）

```
research/{regime}/{direction}/{instrument}/{timeframe}/v{N}_{+/-}{return}%/
├── params.yaml         # 优化参数
├── validation.yaml     # 验证结果
├── attribution.md      # 归因分析
├── train.html          # AlphaForge IS 报告
├── oos.html            # AlphaForge OOS 报告
└── holdout.html        # Portfolio 开封后
```

---

## 2. 策略命名规范

**文件：** `v{N}.py`（同一 timeframe 目录内唯一，N 从 1 递增）

**name 属性：**

```python
name = "{regime}_{direction}_{instrument}_{timeframe}_v{N}"
```

示例：`mild_trend_long_I_daily_v1`、`strong_trend_short_AG_1h_v5`

**Research 目录：** `v{N}_{+/-}{return}%`（OOS 总收益，保留两位小数，正数带 `+`）

品种 ticker 大写：I, AG, RB, HC, J, JM。目录路径使用大写 ticker。

---

## 3. 策略设计要求

### 3.1 基类接口

```python
from strategies.templates.base_strategy import QBaseStrategy

class MyStrategy(QBaseStrategy):
    # === 必填类属性 ===
    name: ClassVar[str] = "mild_trend_long_I_daily_v1"
    regime: ClassVar[str] = "trending"          # "trending" | "mean_reversion"
    horizon: ClassVar[str] = "medium"           # "fast" | "medium" | "slow" | None (MR)
    signal_dimensions: ClassVar[list[str]] = ["momentum", "volume"]
    warmup: ClassVar[int] = 75

    # === 可优化参数（<= 5 个，含 chandelier_mult）===
    fast_period: int = 12
    slow_period: int = 26
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._macd_line, _, _ = macd(self._closes, self.fast_period, self.slow_period, 9)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        # ... 信号逻辑
        return signal

    def get_indicator_config(self) -> list[dict]:
        return [{"name": "MACD", "params": {"fast": self.fast_period, "slow": self.slow_period}}]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [...],
            "subplots": [...],
        }
```

### 3.2 参数约束

- 每策略可优化参数 **<= 5 个**（含 `chandelier_mult`）
- 参数范围窄（2-3x 默认值）
- 风控参数和信号参数一起优化

### 3.3 信号维度多样性

每个 timeframe 目录下的策略应覆盖不同信号维度：

| 类型 | 建议数量 | 信号维度 | 典型指标 |
|------|---------|---------|---------|
| 纯动量 | 4-5 | momentum | TSMOM, EMA Cross, SuperTrend, ADX |
| 动量 + 量价 | 5-6 | momentum + volume/OI | MACD+CMF, SuperTrend+OBV |
| 动量 + 技术 | 4-5 | momentum + technical | MACD+RSI, Aroon+Force |
| 动量 + Carry | 3-4 | momentum + carry | TSMOM+Basis, EMA+Term Structure |
| 多维混合 | 2-3 | 3+ dimensions | MACD+CMF+RSI |

### 3.4 方向约束

| 目录 | 信号范围 | 说明 |
|------|---------|------|
| `long/` | `[0, 1]` | 只做多 |
| `short/` | `[-1, 0]` | 只做空 |
| `mean_reversion/` | `[-1, 1]` | 双向交易 |

基类 `QBaseStrategy` 在 `generate_signals()` 中自动执行方向裁剪。

### 3.5 指标面板（Indicator Panels）

每个策略实现 `get_indicator_panels(datetimes)` 用于 AlphaForge HTML 报告渲染。

- **Overlay（主图叠加）**：价格级指标 — EMA, SuperTrend, Bollinger, Donchian
- **Subplot（独立副图）**：振荡器 — RSI, MACD, ADX, CMF, OBV
- **Signal**：由 `backtest_runner` 自动追加

支持样式：`line`、`step`、`dash`、`bar`、`area`

---

## 4. 新建策略 Checklist

- [ ] 确定 regime / direction / instrument / timeframe
- [ ] 在对应目录创建 `v{N}.py`
- [ ] 继承 `QBaseStrategy`，填写所有必填属性
- [ ] 实现 `on_init_arrays` + `_generate_signal` + `get_indicator_config`
- [ ] 实现 `get_indicator_panels()`
- [ ] 运行优化 → 保存 `params.yaml`
- [ ] 运行验证 → 保存 `validation.yaml`
- [ ] 运行归因 → 保存 `attribution.md`
- [ ] 生成 AlphaForge train.html + oos.html（Industrial 模式，必须传 bar_data）
- [ ] 对照准入标准判定 pass/fail（见 [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)）
- [ ] 更新 summary.yaml

---

## 5. 快速参考

### 关键路径

```
v{N}.py → optimize → params.yaml → validate → validation.yaml
                                             → attribute → attribution.md
                                             → AlphaForge report → train.html + oos.html
                                             → 准入判定 → summary.yaml 更新
```

### Research 产物清单

```
research/{regime}/{direction}/{instrument}/{timeframe}/v{N}_{+/-}{return}%/
├── params.yaml              # 优化参数 + opt_score + is_robust
├── validation.yaml          # 6 层验证结果
├── attribution.md           # 5 层归因分析
├── train.html               # AlphaForge Industrial IS 报告
├── oos.html                 # AlphaForge Industrial OOS 报告
└── holdout.html             # Portfolio 开封后
```

### Timeframe 迁移规则

策略允许迁移到更优 timeframe：移动 `v{N}.py` + `research/.../v{N}_{return}%/` 到目标目录，更新 `name` 属性和 `summary.yaml`。

### 相关文档

| 主题 | 文档 |
|------|------|
| 完整开发流程 | [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) |
| 准入标准 & 禁止事项 | [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) |
| Portfolio 构建 & Regime 激活 | [PORTFOLIO.md](PORTFOLIO.md) |
| AlphaForge V7.6.1 API | [ALPHAFORGE_API.md](ALPHAFORGE_API.md) |
| 系统架构 | [architecture.md](architecture.md) |
