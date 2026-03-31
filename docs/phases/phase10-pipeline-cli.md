# Phase 11: Pipeline + CLI + Reporting

**目标：** 全流程编排自动化，CLI 命令行工具，HTML 报告生成。

**依赖：** 可从 Phase 4 开始逐步搭建，随每个 Phase 完成逐步扩展。

---

## CLI 命令

```bash
# Regime 标注
qbase label I --visualize              # 标注 + 价格叠加图
qbase label I --edit                   # 交互式修改
qbase label I --validate               # 检查完整性

# 策略运行
qbase run trend_medium_v1.py --symbol RB --freq 1h --start 2013
qbase run trend_medium_v1.py --symbol RB --freq 1h --regime strong_trend --direction up

# 优化
qbase optimize trend_medium_v1.py --symbol RB --freq 1h \
    --regime strong_trend --direction up --trials 80

# 验证
qbase validate trend_medium_v1 --regime-cv         # Layer 1
qbase validate trend_medium_v1 --oos               # Layer 2
qbase validate trend_medium_v1 --walk-forward      # Layer 3
qbase validate trend_medium_v1 --dsr               # Layer 4
qbase validate trend_medium_v1 --monte-carlo       # Layer 5
qbase validate trend_medium_v1 --industrial        # Layer 6
qbase validate trend_medium_v1 --all               # 全部 6 层

# 归因
qbase attribute trend_medium_v1 --symbol RB

# Portfolio
qbase portfolio build --symbol RB --regime strong_trend
qbase portfolio score --symbol RB
qbase portfolio report --symbol RB

# 报告
qbase report RB                                    # HTML 报告
```

---

## Pipeline 编排

```python
# pipeline/runner.py
class QBasePipeline:
    """
    全流程编排:
    label → optimize → validate → attribute → portfolio

    支持:
    - 单策略全流程
    - 批量全流程
    - 断点续跑
    """
```

---

## HTML 报告

基于 AlphaForge 的 `HTMLReportGenerator` 扩展：

```
reports/{symbol}/{regime}/
├── strategies/
│   ├── trend_medium_v1.html    # 单策略报告
│   └── ...
├── validation/
│   ├── regime_cv_summary.html  # Regime CV 汇总
│   └── walk_forward.html       # Walk-Forward 结果
├── attribution/
│   ├── trend_medium_v1.html    # 归因报告
│   └── coverage_matrix.html    # Regime 覆盖矩阵
└── portfolio/
    └── portfolio_summary.html  # Portfolio 综合报告
```

---

## 交付物

1. `pipeline/runner.py` — 全流程编排
2. `pipeline/cli.py` — CLI 工具
3. HTML 报告模板 + 生成器
4. 测试覆盖
