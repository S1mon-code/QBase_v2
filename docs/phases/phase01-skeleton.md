# Phase 1: 项目骨架 + 数据基础

**目标：** QBase_v2 repo 可运行，数据流通畅，指标库就位。

**依赖：** 无

---

## 交付物

### 1. 项目目录结构

按 CLAUDE.md 中的目标结构创建全部目录。

### 2. 配置文件

**config/settings.yaml:**
```yaml
alphaforge_path: "/Users/simon/Desktop/alphaforge"
data_dir: "/Users/simon/Desktop/alphaforge/data"
initial_capital: 10_000_000
target_vol: 0.10           # 10% 年化目标波动率
vol_halflife: 60            # Vol Targeting 指数加权半衰期（天）
risk_per_trade: 0.02        # 单笔风险 2%
max_margin_usage: 0.30      # 单策略保证金上限 30%
total_margin_limit: 0.80    # 总保证金上限 80%
```

**config/fundamental_views.yaml（模板）:**
```yaml
views:
  RB: { direction: long, regime: mild_trend }
  HC: { direction: long, regime: mild_trend }
  I:  { direction: short, regime: strong_trend }
  J:  { direction: neutral, regime: mean_reversion }
  JM: { direction: neutral, regime: mean_reversion }
updated_at: "2026-04-01"
updated_by: "基本面团队"
```

### 3. 指标库复制 + Carry 新增

从 v1 复制 `indicators/` 全部 324 个指标，验证 import 正常。

**新增 Carry 指标（indicators/carry/）：**

```python
# indicators/carry/term_structure_carry.py
def term_structure_carry(near_price: np.ndarray, far_price: np.ndarray) -> np.ndarray:
    """期限结构 Carry = (近月 - 远月) / 近月
    > 0: Backwardation（做多有正 carry）
    < 0: Contango（做空有正 carry）
    """
    carry = (near_price - far_price) / near_price
    return carry

# indicators/carry/basis_momentum.py
def basis_momentum(carry: np.ndarray, period: int = 20) -> np.ndarray:
    """Carry 的变化速度 — Carry 在扩大还是缩小"""
    return carry - np.roll(carry, period)

# indicators/carry/roll_yield.py
def roll_yield(close: np.ndarray, settlement: np.ndarray, period: int = 20) -> np.ndarray:
    """换月收益率 — 反映期限结构的实际收益影响"""
    ...
```

### 4. 数据加载验证

确认黑色系 5 品种在 AlphaForge 中可正常加载：

```python
from alphaforge.data.market import MarketDataLoader
loader = MarketDataLoader("data/")

for symbol in ['RB', 'HC', 'I', 'J', 'JM']:
    bars = loader.load(symbol, freq="1h", start="2013-01-01")
    assert len(bars) > 0
    print(f"{symbol}: {len(bars)} bars, {bars._datetime[0]} ~ {bars._datetime[-1]}")
```

**2h 频率验证：** AlphaForge 原生支持 1h，2h 通过 `resample_freqs=["2h"]` 实现。验证 resample 可行性。

### 5. Industrial 模式配置模板

```python
INDUSTRIAL_CONFIG = BacktestConfig(
    initial_capital=10_000_000,
    volume_adaptive_spread=True,
    dynamic_margin=True,
    time_varying_spread=True,
    rollover_window_bars=20,
    asymmetric_impact=True,
    detect_locked_limit=True,
    margin_check_mode="daily",
    margin_call_grace_bars=3,
)
```

### 6. CLAUDE.md + conftest.py + pyproject.toml

---

## 风险

低。纯脚手架，无算法逻辑。唯一风险是 Carry 指标需要近远月合约数据，AlphaForge 当前可能只有主力连续合约。如果近远月数据不可用，Carry 指标先用 settlement vs close 近似。
