# CTA Portfolio Construction Research: QBase_v2 Gap Analysis

**Date:** 2026-04-01
**Objective:** Compare QBase_v2 portfolio construction against professional CTA fund practices and identify actionable gaps.

---

## Executive Summary

QBase_v2 has a solid foundation: Carver-standard signal blending, deflated Sharpe ratio, walk-forward validation, HRP weighting, and a four-level portfolio stop system. However, compared to professional CTAs (Man AHL, Winton, AQR, CFM), the system has **five critical gaps**:

1. **Single-instrument concentration risk** — the single largest divergence from industry practice
2. **Static risk allocation** — no dynamic adjustment of strategy weights based on regime/volatility
3. **No portfolio-level volatility targeting** — only strategy-level vol scaling exists
4. **Greedy selection bias** — forward selection maximizing return is prone to overfitting vs. HRP/NCO
5. **No strategy decay monitoring** — no live detection of alpha erosion

---

## 1. Portfolio Construction Process Comparison

### How Top CTAs Select Strategies

| Firm | Approach | Key Insight |
|------|----------|-------------|
| **Man AHL** | "Cooking up Sharpe" Z-shift framework: diversification + capital efficiency + active risk management. Maximum diversification principle — maximize the number of independent risk sources. Use shrinkage estimators for correlation matrices. | Portfolio construction IS alpha, not just signal generation |
| **Winton** | Research-driven: design strategies from scientific hypotheses, then combine via risk parity across independent signal families. Heavy emphasis on statistical significance before inclusion. | Hypothesis-driven, not data-mined |
| **AQR** | Factor decomposition: momentum, carry, value, defensive. Risk parity allocation across factors, then across instruments within each factor. Enhanced portfolio optimization with robust estimators. | Factor-first, instrument-second |
| **CFM** | "Portfolio Construction Matters" — showed that portfolio construction methodology can change a strategy's Sharpe by 0.3-0.5. Use constrained optimization with transaction cost penalties. Research on strategy decay (Falck, Rej, Thesmar 2021). | Construction methodology matters as much as signal quality |

### QBase_v2 vs. Professional Practice

| Dimension | QBase_v2 | Professional CTAs | Gap Severity |
|-----------|----------|-------------------|--------------|
| Strategy selection | Greedy forward selection maximizing return with Sharpe >= 1.0 | HRP, NCO, or constrained mean-variance with robust estimators | **HIGH** |
| Diversification objective | Correlation threshold 0.5 (hard cutoff) | Continuous: maximize effective N (number of independent bets) | **MEDIUM** |
| Weight calculation | 3-stage: Equal -> InvVol -> HRP x Alpha x Consistency | Risk parity (equal risk contribution) or HRP with shrinkage | **LOW** (HRP already implemented) |
| Return-vs-diversification tradeoff | Maximize absolute return subject to Sharpe constraint | Maximize risk-adjusted return per unit of concentration | **MEDIUM** |
| Number of trials adjustment | Deflated Sharpe Ratio implemented | DSR + Probability of Backtest Overfitting (PBO/CSCV) | **LOW** (DSR exists) |

### Specific Recommendation: Replace Greedy Selection

**Current (QBase_v2):** Greedy forward selection adding strategies that maximize absolute return with Sharpe >= 1.0 and correlation < 0.5.

**Problem:** Greedy selection is a local optimizer. It:
- Is path-dependent (order of evaluation matters)
- Does not consider the full correlation structure simultaneously
- Optimizes return, not risk-adjusted return per unit of concentration
- Has no penalty for model complexity (number of strategies)

**Recommended replacement — Nested Clustered Optimization (NCO):**
From Lopez de Prado (2016) "Building Diversified Portfolios that Outperform Out-of-Sample":

1. **Cluster** strategy returns using hierarchical clustering on the correlation matrix
2. **Intra-cluster allocation** — apply mean-variance (or risk parity) within each cluster
3. **Inter-cluster allocation** — treat each cluster as a single asset, apply mean-variance across clusters
4. **Result:** Reduces Markowitz instability by operating on smaller, better-conditioned matrices

**Implementation priority: HIGH** — this is a code change in `portfolio/selection.py` and `portfolio/weights.py`.

---

## 2. Risk Management at Portfolio Level

### What Professional CTAs Do

Based on Kaminski (2015, Campbell & Company) and Man AHL's dynamic risk management framework:

#### 2a. Portfolio-Level Volatility Targeting (MISSING)

**What it is:** A single volatility target for the entire portfolio, separate from strategy-level vol scaling.

**Formula:** `Portfolio_Leverage_t = Target_Portfolio_Vol / Forecast_Portfolio_Vol_t`

**Why it matters:**
- QBase_v2 has strategy-level vol targeting in `risk/vol_targeting.py` (target_vol / realized_vol per strategy)
- But it has NO portfolio-level vol target that accounts for cross-strategy correlations
- When correlations spike (crisis), individual strategy vol targets don't prevent portfolio vol from exploding
- Professional CTAs run portfolio vol targeting as a SEPARATE layer on top of strategy vol targeting

**Current gap in code:** `risk/vol_targeting.py` only operates on single-strategy returns. No function takes the portfolio covariance matrix as input.

**Implementation:**
```python
def portfolio_vol_target(
    strategy_positions: dict[str, float],
    strategy_returns: np.ndarray,  # T x N matrix
    target_vol: float = 0.20,
    halflife: int = 60,
) -> float:
    """Returns a scalar multiplier for ALL positions to hit target portfolio vol."""
    cov = ewma_covariance(strategy_returns, halflife)
    w = np.array(list(strategy_positions.values()))
    portfolio_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
    return min(target_vol / portfolio_vol, 2.0)  # cap leverage at 2x
```

#### 2b. Dynamic Risk Allocation (PARTIALLY MISSING)

**What it is:** Adjusting strategy weights based on current regime, not just activating/deactivating strategies.

**Current state:** `portfolio/regime_allocator.py` only returns a binary multiplier (0.5 for crisis, 1.0 otherwise). This is crude.

**What CTAs do:**
- **Continuous regime scoring** — not binary. Regime confidence from 0.0 to 1.0.
- **Regime-conditional correlation matrices** — use different correlation estimates per regime
- **Tilt weights toward strategies that historically perform best in the current regime**
- **Reduce gross exposure in uncertain/transitional regimes**

**Specific improvements needed:**
1. Regime confidence score (not binary)
2. Regime-conditional covariance matrix in HRP
3. Regime transition buffer (smooth weight changes over 3-5 days)

#### 2c. Correlation Regime Monitoring (MISSING)

**What it is:** Real-time tracking of cross-strategy correlations vs. historical norms, with alerts when diversification breaks down.

**Why critical:** In crisis periods, correlations between strategies can spike from 0.3 to 0.8+. Your correlation threshold of 0.5 is an in-sample number. Out-of-sample, this threshold will be violated.

**Implementation needed:**
```python
@dataclass
class CorrelationRegime:
    current_avg_corr: float      # rolling 20-day average pairwise correlation
    historical_avg_corr: float   # long-run average
    z_score: float               # how many std devs above normal
    alert_level: str             # "normal", "elevated", "crisis"
```

**Action triggers:**
- z_score > 1.5 -> reduce gross exposure by 20%
- z_score > 2.5 -> reduce gross exposure by 50%
- z_score > 3.0 -> circuit breaker consideration

#### 2d. Tail Risk / Left-Tail Hedging (MISSING)

**What CTAs do:**
- Monitor portfolio skewness and kurtosis in real-time
- Some allocate 2-5% of risk budget to explicit tail hedges (OTM puts, VIX calls)
- For single-instrument systems: tighten stops when implied vol is elevated
- Use Expected Shortfall (CVaR) not just VaR

**For QBase_v2 (single instrument, no options):**
- Compute rolling CVaR (Expected Shortfall at 95% and 99%)
- When CVaR exceeds 2x historical average -> reduce position by 30%
- Implement portfolio-level skewness monitoring

#### 2e. Maximum Leverage / Gross Exposure Limits (PARTIALLY EXISTS)

**Current:** `vol_scale` in `risk/vol_targeting.py` clips to [0.2, 3.0]. Portfolio stops exist.

**What's missing:**
- No explicit gross notional exposure limit relative to NAV
- No net exposure limit
- No per-timeframe exposure limit (daily vs. 1h positions are additive — 70% + 30% = 100%, but can spike higher)

**Recommendation:** Add hard cap at 150% gross exposure (1.5x NAV), with soft warning at 120%.

#### 2f. Drawdown-Based Position Scaling (EXISTS but basic)

**Current:** `portfolio/stops.py` has 4 levels: warning -10%, reduce -15%, circuit -20%, daily -5%.

**What's better (industry practice — "Calmar scaling"):**
- Continuous, not stepped
- `scale = max(0.1, 1.0 - (current_dd / max_allowed_dd)^2)`
- Gradual deleverage starting from first dollar of drawdown
- Faster recovery protocol: re-lever at 50% of de-lever speed

---

## 3. Walk-Forward Portfolio Validation

### What QBase_v2 Has
- Walk-forward validation per strategy (`validation/walk_forward.py`)
- Deflated Sharpe Ratio (`validation/deflated_sharpe.py`)
- Monte Carlo simulation (`validation/monte_carlo.py`)
- Permutation test (`validation/permutation_test.py`)
- Regime cross-validation (`validation/regime_cv.py`)
- Stress testing (`validation/stress_test.py`)
- Industrial check (`validation/industrial_check.py`)

This is a strong validation stack. The validation pipeline is probably the strongest part of the system.

### What's Missing

#### 3a. Walk-Forward PORTFOLIO Construction (CRITICAL GAP)

**Problem:** Walk-forward currently validates individual strategies, but NOT the portfolio construction process itself.

**What should happen:**
1. At each walk-forward window boundary:
   - Run strategy selection using only IS data
   - Compute weights using only IS data
   - Evaluate the SELECTED PORTFOLIO on OOS data
2. Track: Does the portfolio selection remain stable? Do the same strategies keep getting selected?
3. Compute **Portfolio Selection Instability Index**: fraction of strategies that change between adjacent windows

**This is the single most important validation gap.** You validate strategies individually, but the COMBINATION is not validated walk-forward.

#### 3b. Bootstrap Confidence Intervals on Portfolio Metrics (MISSING)

**Implementation:**
```python
def bootstrap_portfolio_sharpe(
    portfolio_returns: np.ndarray,
    n_bootstrap: int = 10000,
    block_size: int = 20,  # block bootstrap preserves autocorrelation
) -> tuple[float, float, float]:
    """Returns (mean, 5th percentile, 95th percentile) Sharpe."""
```

Use **circular block bootstrap** (Politis & Romano) to preserve time-series structure.

**Key question to answer:** Is the portfolio Sharpe of 1.921 statistically significant? What is the 95% CI? If the lower bound is below 1.0, the portfolio is not as robust as it appears.

#### 3c. Probability of Backtest Overfitting — CSCV (MISSING)

From Bailey, Borwein, Lopez de Prado & Zhu (2014):
- Split backtest into S sub-matrices using combinatorial symmetric cross-validation
- For each combination, build portfolio on IS, evaluate on OOS
- **PBO = fraction of combinations where OOS rank of IS-optimal strategy is below median**
- PBO > 0.5 means the selection is likely overfit

**This directly tests whether your greedy forward selection is overfit.**

#### 3d. Strategy Selection Stability Over Time (MISSING)

Track across rolling windows:
- Which strategies get selected?
- How stable are the weights?
- Compute **Jaccard similarity** between adjacent window strategy sets
- If Jaccard < 0.5 (less than half strategies persist), the selection is unstable = likely overfit

---

## 4. Rebalancing and Maintenance

### Professional CTA Practice

| Aspect | Industry Practice | QBase_v2 |
|--------|------------------|----------|
| **Rebalancing frequency** | Monthly weights review, daily risk scaling | Daily continuous rebalancing (daily), fixed entry (1h) |
| **Strategy addition** | Quarterly review cycle, 6-12 month OOS track record required | 13 hard filters (good) |
| **Strategy removal** | Bayesian changepoint detection, 3-6 month underperformance window | No formal removal protocol |
| **Performance monitoring** | Real-time: PnL attribution, factor exposure decomposition, correlation monitoring | Not implemented |
| **Early warning** | Rolling Sharpe z-score, drawdown velocity, correlation spike detection | Portfolio stops only |
| **Portfolio shutdown** | Predefined criteria: max DD breach, Sharpe < 0 over 12 months, regulatory | No shutdown protocol |

### Specific Missing Pieces

#### 4a. Strategy Retirement Protocol (MISSING)

**When to remove a strategy from live portfolio:**

1. **Performance decay detection** (CFM's research: "Why and how systematic strategies decay"):
   - Rolling 6-month Sharpe drops below 0.0 for 2 consecutive months
   - Cumulative OOS return turns negative
   - Strategy's marginal contribution to portfolio Sharpe turns negative

2. **Bayesian changepoint detection** (Quant Beckman's "Switch-Off"):
   - Monitor each strategy's return stream for structural breaks
   - When changepoint probability > 0.8, put strategy on "watch"
   - When probability > 0.95 for 20+ days, remove from portfolio

3. **Correlation breakdown:**
   - Strategy's correlation with portfolio exceeds 0.7 (was < 0.5 at admission)
   - Strategy provides negative diversification benefit

#### 4b. Real-Time Performance Attribution (MISSING)

Break down daily PnL into:
- Signal alpha (did the forecasts add value?)
- Regime allocation (did we size correctly for the regime?)
- Risk management drag (cost of stops, vol scaling, position limits)
- Transaction costs
- Timing (did entry/exit timing add or subtract value?)

#### 4c. Rebalancing Protocol

**Current:** Daily continuous rebalancing for daily strategies. This is fine.

**CFA Institute research (Benhamou et al. 2026):** CTA allocations by trend horizon show:
- Fast signals (< 1 month): daily rebalancing essential
- Medium signals (1-6 months): weekly rebalancing sufficient
- Slow signals (6-12 months): monthly rebalancing with daily risk overlay

**Recommendation for QBase_v2:**
- Keep daily rebalancing for 1h strategies
- Consider weekly weight rebalancing for daily strategies (reduce turnover)
- Daily risk overlay (vol targeting, stops) regardless

---

## 5. Cost and Capacity Analysis

### Transaction Cost Estimation (MISSING)

**What CTAs model:**
1. **Spread cost:** Half-spread per trade (for Iron Ore futures: typically 0.5-1 tick)
2. **Market impact:** Proportional to `sqrt(volume) / ADV` — significant for concentrated single-instrument trading
3. **Slippage:** Gap between signal price and execution price
4. **Opportunity cost:** Cost of NOT trading when signal fires but risk limits prevent execution

**For QBase_v2:**
- 60 strategies on single instrument, 2 timeframes
- 1h strategies generate ~30-50 round-trips per strategy per year
- Daily strategies generate ~10-20 round-trips per strategy per year
- With 8 selected strategies: ~150-300 total round-trips/year
- **Estimated cost:** 0.5-1.5% annual drag (needs explicit modeling)

**Implementation needed:**
```python
@dataclass
class TransactionCostModel:
    spread_ticks: float = 1.0      # half-spread in ticks
    tick_value: float = 100.0       # RMB per tick for Iron Ore
    impact_coefficient: float = 0.1 # temporary impact
    delay_bars: int = 1             # execution delay
```

### Capacity Constraints for Single-Instrument Portfolio (CRITICAL)

**This is the elephant in the room.**

Professional CTAs trade 50-200+ instruments precisely because single-instrument capacity is limited:
- Iron Ore futures (DCE): daily volume ~1M contracts, but effective capacity for a systematic trader is ~0.5-1% of daily volume
- At ~500 RMB/contract, ~1000 contracts/day max comfortable = ~500K RMB daily risk capacity
- With 20% target vol, this supports ~2.5M RMB NAV comfortably
- Beyond this, market impact erodes returns

**Key insight from Carver:** "The single most important thing you can do to improve a systematic trading system is add more instruments." He runs 100+ instruments specifically for this reason.

**Recommendation:** Phase 9 (Expand Instruments) should be elevated to higher priority. Even adding 2-3 correlated instruments (Rebar RB, Hot-Rolled Coil HC, Coke J) provides meaningful capacity expansion and marginal diversification.

---

## 6. What Top CTAs Do That We Don't

### 6a. Cross-Instrument Diversification (NOT IN SYSTEM — CRITICAL)

**Industry standard:** Man AHL trades 400+ markets. Winton trades 100+. Even small CTAs trade 20-50 instruments.

**Impact on Sharpe:** Adding uncorrelated instruments provides a "free lunch":
- `Portfolio_Sharpe ≈ Single_Sharpe × sqrt(N_effective)` where N_effective accounts for correlations
- With 5 instruments at avg correlation 0.3: N_effective ≈ 3.3, Sharpe multiplier ≈ 1.8x
- Your current Sharpe of 1.921 on a single instrument is impressive, but fragile

**Carver's Instrument Diversification Multiplier (IDM):**
- For 1 instrument: IDM = 1.0
- For 5 instruments (avg corr 0.3): IDM ≈ 1.8
- For 20 instruments (avg corr 0.2): IDM ≈ 3.5

Your system has IDM = 1.0 by definition. This is the single biggest opportunity for improvement.

### 6b. Cross-Sector Allocation (NOT IN SYSTEM)

CTAs allocate across:
- **Commodities** (metals, energy, agriculture) — 30-40% typical
- **Fixed income** (bonds, rates) — 20-30%
- **Equities** (index futures) — 15-25%
- **FX** — 15-25%

Correlation between sectors is typically 0.1-0.3, providing massive diversification.

### 6c. Factor Decomposition at Portfolio Level (NOT IN SYSTEM)

AQR and CFM decompose returns into:
- **Momentum factor:** Are we long things going up, short things going down?
- **Carry factor:** Are we long positive carry, short negative carry?
- **Value factor:** Are we buying cheap, selling expensive?
- **Defensive factor:** Are we positioned for risk-off?

**For single-instrument QBase_v2:** This is less relevant but still useful:
- Decompose portfolio return into: trend-following alpha, mean-reversion alpha, carry alpha
- Monitor factor balance — if 80% of return comes from trend and trend regime ends, portfolio is vulnerable

### 6d. Dynamic Leverage Management (PARTIALLY MISSING)

**Current:** Static 70/30 daily/1h split. Vol targeting per strategy. No portfolio-level leverage management.

**Industry practice:**
- Target portfolio vol (e.g., 15% annualized)
- Dynamically adjust gross leverage to maintain this target
- Reduce leverage in high-vol environments (opposite of what constant-vol targeting does at strategy level)
- Use **Conditional Volatility Targeting** (Bongaerts, Kang, Van Dijk 2020): condition on current vol regime, not just current vol level

### 6e. Portfolio-Level Volatility Scaling (NOT IN SYSTEM)

**This is different from strategy-level vol targeting.**

Strategy-level (what you have): Each strategy scales its own position by target_vol / realized_vol.
Portfolio-level (what you need): After combining all strategy positions, scale the aggregate by portfolio_target_vol / portfolio_realized_vol.

**Why both are needed:**
- Strategy-level vol targeting doesn't account for diversification benefit
- When strategies are diversified, portfolio vol < sum of strategy vols
- Portfolio-level scaling captures this, allowing higher per-strategy leverage
- When correlations spike, portfolio-level scaling automatically delevers

---

## 7. Common Mistakes in CTA Portfolio Construction

### 7a. Overfitting in Strategy Selection (RISK: MEDIUM-HIGH)

**Your risk factors:**
- 60 strategies tested, 8 selected — selection ratio of 13% is reasonable
- BUT: greedy forward selection maximizing return is a known overfitting pathway
- Deflated Sharpe ratio mitigates this somewhat, but DSR adjusts for the number of trials, not for the selection method itself

**CFM's research (Falck et al. 2021):** Found that strategy decay is predicted by:
1. **Year of publication** — every year, newly published factor Sharpe decays by 5 percentage points more
2. **Signal complexity** — more operations to calculate = more overfitting risk
3. **Sensitivity to outliers** — if Sharpe is driven by a few extreme observations, decay is faster

**Mitigation for QBase_v2:**
- Replace greedy selection with NCO/HRP-based selection
- Implement CSCV (Probability of Backtest Overfitting)
- Track which strategies are selected across rolling windows
- Penalize complex strategies (more parameters = higher bar for admission)

### 7b. Look-Ahead Bias in Correlation Estimation (RISK: HIGH)

**The problem:** Your correlation threshold of 0.5 uses the FULL sample correlation. But when building the portfolio at time T, you should only know correlations up to time T.

**Fix:**
- Use rolling window correlations (e.g., 252-day) in walk-forward selection
- Apply shrinkage (Ledoit-Wolf) — you already have this in `weights.py`, good
- Test correlation stability: compute correlation in first half vs. second half of sample
- If difference > 0.2, the correlation estimate is unreliable

### 7c. Survivorship Bias in Backtest (RISK: LOW-MEDIUM)

**For strategy-level:** All 60 strategies were developed on the same data — potential for collective survivorship bias.

**Mitigation:**
- Ensure that strategies were not tuned on the OOS period
- Use expanding window walk-forward (already have this)
- Track the strategies that were REJECTED and verify they actually underperform OOS

### 7d. Excessive Complexity (RISK: LOW)

**Current state:** 13 hard filters, 5-dimensional optimization objective, 6-layer validation — this is actually well-structured, not excessively complex.

**One concern:** The 5-dimensional optimization function (Performance 40% + Significance 15% + Consistency 15% + Risk 15% + Alpha 15%) introduces 5 hyperparameters (the weights themselves). These weights could be a source of overfitting.

**Recommendation:** Test sensitivity: vary each weight by +/- 10% and check if the selected portfolio changes significantly.

### 7e. Ignoring Costs in Portfolio Optimization (RISK: MEDIUM)

**Current:** No transaction cost model in strategy selection or portfolio optimization.

**Impact:** Strategies with high turnover (especially 1h strategies) may look great gross of costs but underperform net. The 1h strategies generating 30-50 round-trips/year could have 1-2% annual cost drag.

**Fix from Carver:** Penalize turnover in the optimization: `net_sharpe = gross_sharpe - cost_penalty * turnover`

---

## 8. Academic References

### Primary References (Must-Read)

| Paper | Key Contribution | Relevance to QBase_v2 |
|-------|-----------------|----------------------|
| **Lopez de Prado (2016)** "Building Diversified Portfolios that Outperform OOS" | NCO algorithm — cluster, then optimize within/across clusters | Replace greedy selection |
| **Bailey & Lopez de Prado (2014)** "The Deflated Sharpe Ratio" | Adjust Sharpe for multiple testing | Already implemented |
| **Bailey et al. (2014)** "Probability of Backtest Overfitting" | CSCV method to quantify PBO | Add to validation pipeline |
| **Harvey, Liu & Zhu (2016)** "...and the Cross-Section of Expected Returns" | Multiple testing framework: t-stat > 3.0 needed for new factors | Apply to strategy admission |
| **Carver (2015)** "Systematic Trading" | FDM, forecast scaling, instrument diversification, portfolio of trading rules | Foundation of current system |
| **Falck, Rej & Thesmar (2021, CFM)** "Why and How Systematic Strategies Decay" | Strategy complexity predicts OOS decay; publication year matters | Strategy retirement protocol |
| **Kaminski (2015, Campbell)** "Quantifying CTA Risk Management" | Factor framework for CTA risk; position sizing = conviction x allocation / vol | Validate risk framework |
| **Bongaerts et al. (2020)** "Conditional Volatility Targeting" | Condition vol target on regime, not just current vol level | Improve vol targeting |
| **Baltas & Kosowski (2013, CME)** "Improving TSMOM Strategies" | Vol estimator choice matters; use exponential vs. simple | Validate vol estimator |

### Supplementary References

| Paper | Key Contribution |
|-------|-----------------|
| **Benhamou et al. (2026, CFA Institute)** "Decoding CTA Allocations by Trend Horizon" | Allocation varies significantly by horizon; fast/medium/slow decomposition |
| **Baltas (2026, Return Stacked)** "Trend-Following, Risk-Parity and the Influence of Correlations" | Risk parity outperforms when cross-strategy correlations are low |
| **Man AHL (2025)** "Cooking up Sharpe: Recipe for Portfolio Construction" | Z-shift framework: diversification + capital efficiency + risk management |
| **Man AHL (2016)** "Maximum Diversification" | Maximize effective number of independent bets |
| **Lopez de Prado (2016)** "Building Diversified Portfolios That Outperform OOS (Slides)" | HRP algorithm with step-by-step visual explanation |

---

## 9. Prioritized Action Plan

### Tier 1: Critical (Do Before Going Live)

| # | Action | Effort | Impact | Files Affected |
|---|--------|--------|--------|----------------|
| 1 | **Walk-forward portfolio construction validation** | 3-5 days | Prevents deploying an overfit portfolio | `validation/walk_forward.py`, new `validation/portfolio_wf.py` |
| 2 | **Portfolio-level volatility targeting** | 2-3 days | Prevents vol explosion when correlations spike | new `risk/portfolio_vol_target.py` |
| 3 | **Transaction cost model** | 2-3 days | Ensures net-of-cost profitability | new `risk/transaction_costs.py`, modify `portfolio/selection.py` |
| 4 | **Correlation regime monitoring** | 2-3 days | Early warning for diversification breakdown | new `monitoring/correlation_monitor.py` |
| 5 | **Replace greedy selection with NCO** | 3-5 days | Reduces overfitting in strategy selection | `portfolio/selection.py`, `portfolio/weights.py` |

### Tier 2: High Priority (First Month of Live Trading)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 6 | **CSCV (Probability of Backtest Overfitting)** | 2-3 days | Quantifies selection overfitting risk |
| 7 | **Strategy retirement protocol** | 2-3 days | Prevents dead strategies from eroding returns |
| 8 | **Bootstrap confidence intervals** | 1-2 days | Quantifies uncertainty in portfolio Sharpe |
| 9 | **Continuous drawdown scaling** (replace stepped stops) | 1-2 days | Smoother risk management |
| 10 | **Gross/net exposure limits** | 1 day | Hard safety guardrail |

### Tier 3: Strategic (First Quarter)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 11 | **Expand to 3-5 instruments** (Phase 9) | 2-3 weeks | Biggest Sharpe improvement available |
| 12 | **Dynamic regime-conditional weights** | 1 week | Better adaptation to changing regimes |
| 13 | **Real-time PnL attribution** | 1 week | Understand what's driving returns |
| 14 | **CVaR-based tail risk monitoring** | 2-3 days | Better tail risk measurement than VaR |
| 15 | **Rebalancing cost optimization** | 2-3 days | Reduce unnecessary turnover |

---

## 10. Key Takeaways

### What QBase_v2 Does Well (Relative to Industry)
1. **Carver-standard signal blending** — forecast scaling, FDM, capping ±20. This is exactly what Man AHL uses.
2. **Multi-layer validation** — 6 validation layers including DSR, walk-forward, Monte Carlo, permutation testing. More thorough than most retail systems.
3. **Regime-aware strategy activation** — fundamental view + technical regime matching. This is a genuine edge.
4. **HRP weighting** (Stage 3 in weights.py) — this is the state-of-the-art from Lopez de Prado.
5. **Portfolio stops** — 4-level system is standard professional practice.
6. **Hard admission filters** — 13 filters is rigorous.

### What QBase_v2 Is Missing (Critical Gaps)
1. **Portfolio-level vol targeting** — the single most impactful missing risk layer
2. **Walk-forward portfolio validation** — validates strategies individually but not the portfolio construction itself
3. **Transaction cost model** — strategies are evaluated gross of costs
4. **Correlation regime monitoring** — static correlation threshold, no dynamic monitoring
5. **Strategy decay detection** — no protocol for identifying and removing dying strategies
6. **Cross-instrument diversification** — the structural limitation that caps risk-adjusted returns

### The Bottom Line

QBase_v2's signal generation and validation infrastructure is strong. The primary gaps are in **portfolio-level risk management** (vol targeting, correlation monitoring, dynamic leverage) and **portfolio-level validation** (walk-forward portfolio construction, CSCV). These are the areas where professional CTAs invest the most engineering effort, and where the largest marginal improvements are available.

The single most impactful long-term improvement is **instrument diversification** (Phase 9). Everything else provides incremental improvement; adding instruments provides multiplicative improvement to risk-adjusted returns.

---

## Sources

- [Man AHL — Cooking up Sharpe: A Recipe for Portfolio Construction (2025)](https://www.man.com/insights/cooking-up-sharpe)
- [Man AHL — Maximum Diversification](https://www.man.com/insights/ahl-explains-maximum-diversification)
- [Man AHL — Dynamic Risk Management](https://www.man.com/it/insights/applying-dynamic-risk-management)
- [Kaminski (2015) — Quantifying CTA Risk Management (CME/Campbell)](https://www.cmegroup.com/education/files/quantifying-cta-risk-Management.pdf)
- [Lopez de Prado (2016) — Building Diversified Portfolios that Outperform OOS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)
- [Bailey & Lopez de Prado (2014) — The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Bailey et al. (2014) — Probability of Backtest Overfitting](https://www.quantbeckman.com/api/v1/file/b9d11388-8c7c-40ca-a072-b8eed41505ae.pdf)
- [Harvey, Liu & Zhu (2016) — ...and the Cross-Section of Expected Returns](https://academic.oup.com/rfs/article/29/1/5/1843824)
- [Falck, Rej & Thesmar (2021, CFM) — Why and How Systematic Strategies Decay](https://www.cfm.com/wp-content/uploads/2022/12/312-2021-05-Why-and-how-systematic-strategies-decay.pdf)
- [CFM — Portfolio Construction Matters](https://cfm.com/portfolio-construction-matters/)
- [CFM — A Systematic Path to True Diversification](https://www.cfm.com/a-systematic-path-to-true-diversification/)
- [AQR — Understanding Risk Parity](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Understanding-Risk-Parity.pdf)
- [AQR — Enhanced Portfolio Optimization](https://www.aqr.com/Insights/Research/White-Papers/Enhanced-Portfolio-Optimization)
- [Benhamou et al. (2026, CFA Institute) — Decoding CTA Allocations by Trend Horizon](https://rpc.cfainstitute.org/blogs/enterprising-investor/2026/decoding-cta-allocations-by-trend-horizon)
- [Baltas (2026, Return Stacked) — Trend-Following, Risk-Parity and Correlations](https://www.returnstacked.com/academic-review/trend-following-risk-parity-and-the-influence-of-correlations/)
- [Bongaerts et al. (2020) — Conditional Volatility Targeting](https://repub.eur.nl/pub/130215/Bongaerts-Kang-van-Dijk-Conditional-volatility-targeting-2020-FAJ.pdf)
- [Breaking Alpha — Portfolio-Level Risk Constraints for Multi-Strategy Algorithms](https://breakingalpha.io/insights/portfolio-level-risk-constraints.html)
- [Carver — Clustering Trading Rule P&L](https://qoppac.blogspot.com/2023/05/clustering-trading-rule-p.html)
- [Carver — Instrument Diversification](https://qoppac.blogspot.com/2023/03/i-got-more-than-99-instruments-in-my.html)
- [Carver — Fit Forecast Weights](https://qoppac.blogspot.com/2021/05/fit-forecast-weights-by-instrument-by.html)
- [Quant Beckman — Bayesian Changepoint Detection for Strategy De-allocation](https://www.quantbeckman.com/p/with-code-switch-off-bayesian-online)
- [Deep (2025, arxiv) — Walk-Forward Validation Framework for Market Microstructure Signals](https://arxiv.org/html/2512.12924v1)
- [Lopez de Prado — HRP and NCO Innovations](https://www.quantresearch.org/Innovations.htm)
- [Winton — What We Do](https://www.winton.com/about)
