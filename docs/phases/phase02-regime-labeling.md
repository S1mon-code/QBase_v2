# Phase 2: Regime 标注系统 + 数据分割

**目标：** 对黑色系品种历史行情自动初标 + 人工校正，输出标注 YAML，同时完成 train/oos/holdout 分割。

**依赖：** Phase 1（数据加载可用）

---

## 2 类 Regime（v2.5 简化）

| Regime | 基本面语言 | 标注方法 | 策略池 |
|--------|-----------|---------|--------|
| **Long** | 上涨趋势（强/温和） | Bry-Boschan + R² 过滤 | 趋势跟踪（做多） |
| **Short** | 下跌趋势（强/温和） | Bry-Boschan + R² 过滤 | 趋势跟踪（做空） |

> **v2.5 变更：** 从 4 类 (strong_trend/mild_trend/mean_reversion/crisis) 简化为 2 类 (long/short)，±1 月 buffer，6:3:1 分割。

---

## 量化初筛器（Bry-Boschan 变体）

```python
# regime/labeler.py
def auto_label(prices, config) -> List[RegimeLabel]:
    """
    1. 识别局部 Peak/Trough（前后 N 个月最高/最低）
    2. 计算相邻极值间涨跌幅
    3. 按方向分类 (v2.5):
       - 上涨段 → Long
       - 下跌段 → Short
       - R² 过滤弱趋势段
    4. 最小持续期: 1 个月
    5. 前后各加 buffer_months 个月
    """
```

## 阈值配置

```yaml
# config/regime_thresholds.yaml (v2.5 简化)
default:
  min_duration_months: 1
  buffer_months: 1             # v2.5: ±1月 buffer
  r2_filter_threshold: 0.3     # R² 过滤弱趋势
  split_ratio: [6, 3, 1]       # train/oos/holdout = 6:3:1
```

---

## 标注 YAML 格式

```yaml
# data/regime_labels/I.yaml
instrument: I
version: 1
labeled_by: "auto + manual review"
labels:
  - start: "2015-06-01"
    end: "2016-02-28"
    regime: long
    direction: up
    driver: "供给侧改革"
    buffer_start: "2015-05-01"
    buffer_end: "2016-03-31"
    split: train                # ← 标注时即分好

  - start: "2020-11-01"
    end: "2021-05-31"
    regime: long
    direction: up
    split: oos

  - start: "2024-01-01"
    end: "2024-08-31"
    regime: long
    direction: up
    split: holdout
```

---

## 数据分割（标注时完成）

每种 regime × direction 的时段做三段式分割：

```
训练集 (60%): 优化器用
OOS (20%):    验证用
Holdout (20%): 最终确认，只跑一次
```

**规则：**
- 分割一经确定，终身不变
- Holdout 在整个开发过程中不能碰
- 每个 split 应包含不同市场环境（不能全是牛市）

---

## 标注可视化 + 人工校正

```bash
qbase label I --visualize          # 生成标注叠加价格图
qbase label I --edit               # 交互式修改
qbase label I --validate           # 检查标注完整性
```

---

## 数据增量更新

新数据到来后：
1. 自动标注器对新时段生成建议标注
2. 人工审核确认
3. 新时段默认归入 train split（不动已有的 oos/holdout）

---

## 交付物

1. `regime/labeler.py` — 自动标注器
2. `regime/schema.py` — 标注 YAML schema + 读写
3. `regime/visualizer.py` — 标注可视化
4. `regime/matcher.py` — 按 regime + direction 筛选时段
5. `config/regime_thresholds.yaml` — 阈值配置
6. 黑色系 5 品种历史标注初版（自动 + 人工校正）
7. 测试覆盖

---

## 风险

中。阈值需要反复调试才能匹配基本面团队直觉。人工校正工作量较大（5 品种 × ~13 年数据）。
