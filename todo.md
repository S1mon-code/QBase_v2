# QBase_v2 — TODO

## 当前状态

Phase 1-8 完成。Phase 10、11 进行中。Phase 9 等待真实数据。

---

## Phase 1: 项目骨架 + 数据基础 [完成]

- [x] 创建完整目录结构
- [x] pyproject.toml + conftest.py
- [x] config/settings.yaml
- [x] config/fundamental_views.yaml 模板
- [x] config/regime_thresholds.yaml
- [x] 复制 v1 indicators/ 到 v2
- [x] 检查 indicators 语法和结构，重新整理分类
- [x] 更新 indicators/indicators.md
- [x] 新增 Carry 指标 (term_structure_carry, basis_momentum, roll_yield)
- [x] Industrial 模式配置模板
- [x] CLAUDE.md 最终版

## Phase 2: Regime 标注系统 [完成 90%]

- [x] regime/labeler.py — Bry-Boschan 自动标注器
- [x] regime/schema.py — YAML schema + 读写
- [x] regime/matcher.py — 按 regime+direction 筛选时段
- [ ] regime/visualizer.py — 标注叠加价格图 (pending)
- [ ] RB/HC/I/J/JM 历史标注 — 需要真实数据

## Phase 3: 风控模块 [完成]

- [x] risk/chandelier.py
- [x] risk/vol_targeting.py
- [x] risk/position_sizer.py
- [x] risk/directional_filter.py
- [x] risk/vol_classifier.py
- [x] risk/portfolio_stops.py

## Phase 4: 策略模板 + 第一批策略 [完成 90%]

- [x] strategies/templates/base_strategy.py
- [x] strategies/templates/trending_template.py
- [x] strategies/templates/mean_reversion_template.py
- [x] strategies/baselines/tsmom_fast.py
- [x] strategies/baselines/tsmom_medium.py
- [x] strategies/baselines/tsmom_slow.py
- [x] trend_medium_v1 (SuperTrend + Volume Momentum)
- [x] trend_medium_v2 (ADX + Force Index)
- [x] trend_medium_v3 (MACD + OI Flow)
- [x] trend_medium_v4 (KAMA Crossover + Volume Efficiency)
- [x] trend_medium_v5 (EMA Ribbon + RSI)
- [x] trend_fast_v1
- [x] trend_slow_v1
- [x] mr_v1 (Bollinger + RSI)
- [ ] 裸逻辑验证 (需要真实数据)
- [ ] vs TSMOM Baseline 对比 (需要真实数据)

## Phase 5: 优化器 [完成]

- [x] optimizer/core.py
- [x] optimizer/robustness.py
- [x] optimizer/regime_optimizer.py
- [x] optimizer/param_discovery.py
- [x] optimizer/trial_registry.py
- [x] optimizer/config.py

## Phase 6: 验证体系 [完成]

- [x] validation/regime_cv.py
- [x] validation/oos_validator.py
- [x] validation/walk_forward.py
- [x] validation/deflated_sharpe.py
- [x] validation/monte_carlo.py
- [x] validation/permutation_test.py
- [x] validation/industrial_check.py
- [x] validation/stress_test.py
- [x] validation/pipeline.py

## Phase 7: 归因分析 [完成]

- [x] attribution/signal.py
- [x] attribution/horizon.py
- [x] attribution/regime.py
- [x] attribution/baseline.py
- [x] attribution/operational.py
- [x] attribution/coverage.py
- [x] attribution/decay.py
- [x] attribution/report.py

## Phase 8: Portfolio 构建 [完成]

- [x] portfolio/signal_blender.py
- [x] portfolio/weights.py
- [x] portfolio/regime_allocator.py
- [x] portfolio/constraints.py
- [x] portfolio/selection.py
- [x] portfolio/scorer.py
- [x] portfolio/stops.py
- [x] portfolio/rebalance.py
- [x] portfolio/retirement.py

## Phase 9: 扩展品种 [待实盘数据]

- [ ] HC 全流程 (标注 → 优化 → 验证 → 归因 → Portfolio)
- [ ] I 全流程
- [ ] J 全流程
- [ ] JM 全流程

## Phase 10: Pipeline + CLI + Reporting [进行中 70%]

- [x] pipeline/runner.py
- [x] pipeline/cli.py
- [ ] HTML 报告模板 (待实现)

## Phase 11: 监控 + 实盘部署 [进行中 60%]

- [x] monitoring/dashboard.py
- [x] monitoring/decay_detector.py
- [x] monitoring/regime_alert.py
- [x] monitoring/retirement.py
- [ ] Paper Trading 验证 (待实盘)

---

## 测试覆盖

- 总测试数: 576+ passed (持续增长)
- 覆盖模块: attribution, config, monitoring, optimizer, pipeline, portfolio, regime, risk, strategies, validation
- 运行: `python -m pytest tests/ -v`
