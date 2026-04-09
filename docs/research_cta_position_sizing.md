# CTA Position Sizing: Deep Research Report

> Research compiled from Robert Carver (pysystemtrade), Man AHL, AQR, Campbell & Company,
> Huatai Securities (华泰金工), and academic/practitioner literature.

---

## 1. Single-Strategy Position Sizing Methods

### 1.1 Volatility Targeting (Industry Standard)

**This is THE dominant method used by professional CTAs.** Every major CTA (Man AHL, AQR, Winton, Campbell, Millburn) uses some form of volatility targeting.

#### Core Formula (Robert Carver / pysystemtrade)

```
Position (contracts) = (Forecast / Avg_Forecast) × Volatility_Scalar × Instrument_Weight × IDM

Where:
  Volatility_Scalar = Daily_Cash_Vol_Target / Instrument_Daily_Cash_Vol
  Daily_Cash_Vol_Target = (Capital × Annual_Vol_Target) / 16
  Instrument_Daily_Cash_Vol = Price × Multiplier × Daily_%_Vol
  Daily_%_Vol = Annual_%_Vol / 16
```

**Concrete example:**
- Capital: $1,000,000
- Annual vol target: 25%
- Annual cash vol target: $250,000
- Daily cash vol target: $250,000 / 16 = $15,625
- Oil futures daily vol: $668.33/contract
- Volatility scalar (max contracts at full risk): $15,625 / $668.33 = 23.4 contracts
- With forecast = +10 (average), instrument_weight = 0.1, IDM = 2.5:
  Position = (10/10) × 23.4 × 0.1 × 2.5 = 5.85 contracts ≈ 6 contracts

#### What Vol Target Do Professionals Use?

| Fund Type | Typical Annual Vol Target | Notes |
|-----------|--------------------------|-------|
| Large CTAs (Man AHL, AQR) | 10-15% | Conservative, institutional money |
| Medium CTAs | 15-20% | |
| Aggressive CTAs | 20-30% | |
| Robert Carver personal | 25% | Half-Kelly of SR=1.0 |
| Campbell & Company baseline | ~4% monthly ≈ ~14% annual | Per their research papers |

#### How to Estimate Realized Volatility

| Method | Lookback | Used By |
|--------|----------|---------|
| Simple rolling std × √256 | 25-day | Industry standard, Carver uses 25-day |
| EMA(span=36) of daily returns | ~36 day effective | Carver's preferred (EMA-36) |
| EMA(span=30) for std | 30-day | pysystemtrade risk calculations |
| EWMA (RiskMetrics) | λ=0.94 | Many institutional desks |
| 3-month lookback | ~63 days | Campbell baseline |
| 6-month lookback | ~126 days | Campbell slow-react variant |
| ATR(20) / Price | 20-day | Common for retail/semi-pro |

**Key insight from Campbell & Company research:** They found that the "volatility factor" (difference between slow and fast vol estimation) was *not* significantly beneficial on average. The industry mostly converges on 20-60 day lookbacks.

**Key insight from Harvey et al. (2018, "The Impact of Volatility Targeting"):**
- Vol targeting reduces volatility-of-volatility from 4.6% to 1.8% (US equities)
- Improves Sharpe from 0.40 to 0.48-0.51
- Reduces kurtosis (fat tails)
- Effectively introduces momentum into the strategy (high vol → reduce position → similar to trend following)

### 1.2 Kelly Criterion and Half-Kelly

#### Full Kelly
```
Kelly_fraction = Sharpe_Ratio² / σ²   (for continuous strategies)
≈ Vol_target should equal Sharpe_Ratio  (Carver simplification)
```

#### Carver's Practical Approach
1. Backtest SR = 1.0 → Mark down by 25% → Effective SR = 0.75
2. Kelly vol target = 75% → Half-Kelly = 37.5%
3. **Carver uses 25% as his personal target** (conservative half-Kelly, rounded down)

| Investor Type | Expected SR | Kelly Target | Half-Kelly | Carver Recommendation |
|---------------|-------------|-------------|------------|----------------------|
| Asset allocator (static) | 0.4 | 40% | 20% | 20% max |
| Semi-auto trader | 0.2-0.5 | 20-50% | 10-25% | Max 25% |
| Fully systematic | 0.5-1.0 | 50-100% | 25-50% | Mark down SR 25%, then half-Kelly |
| Negative skew strategies | Any | K | K/4 | Quarter-Kelly |

**Critical warning:** Full Kelly gives ~10% probability of losing half your money over 10 years. Half-Kelly is the practical standard.

### 1.3 Signal-Proportional Sizing (Continuous Forecasting)

This is the Carver/pysystemtrade innovation and is now considered best practice:

```
Position = f(Forecast_strength)

Forecast range: [-20, +20]  (capped)
Average absolute forecast: 10
Forecast = 0  → No position
Forecast = +10 → Average long position
Forecast = +20 → Maximum long (2x average)
Forecast = -10 → Average short position
```

**Why this matters:**
- Rob Carver showed that forecast strength *predicts* future returns (monotonically)
- Expected risk = Target_risk × Relative_forecast_strength × Relative_correlation_factor
- When forecasts are strong (+20), you WANT 2x the risk
- When forecasts are weak (+2), you want minimal risk
- This is fundamentally different from binary long/short systems

**Man AHL also uses continuous signals**, as described in their publications. Their trend-following models output signal strength, not binary directions.

### 1.4 Risk Parity Per Trade

Used when you want each position to contribute equal risk:

```
Weight_i = (1/σ_i) / Σ(1/σ_j)

Or equivalently:
Contracts_i = Risk_budget_i / (Contract_value × σ_daily)
```

**Huatai Securities (华泰金工) approach for Chinese CTA:**
- Build risk parity portfolio of 10 commodity strategies
- Weights = inverse of volatility
- Rebalance every 20 trading days
- Result: Annual return 15.6%, Sharpe 2.045, Max DD 6.6%

---

## 2. Scaling / Pyramiding

### 2.1 Do Top CTAs Pyramid?

**Short answer: No, not in the traditional sense.** Professional CTAs do NOT use discrete pyramiding (adding fixed lots at breakout levels). Instead, they use **continuous position adjustment**:

#### Carver/Professional Approach: Continuous Forecast-Based Position Sizing
```
Day 1: Forecast = +5  → Position = 3 contracts
Day 5: Forecast = +10 → Position = 6 contracts  (added 3)
Day 12: Forecast = +18 → Position = 11 contracts (added 5)
Day 20: Forecast = +12 → Position = 7 contracts  (reduced 4)
Day 30: Forecast = +3  → Position = 2 contracts  (reduced 5)
```

This IS scaling in/scaling out, but driven by signal strength rather than price levels.

### 2.2 Position Inertia (Buffering)

**Critical concept:** Don't trade every tiny change in target position.

Carver's rule: **If existing position is within 10% of target, don't trade.**

```
Target = 10 contracts
Current = 9 contracts
Buffer = 10% of 10 = 1 contract
Range = [9, 11]
Since 9 is within [9, 11], DON'T TRADE.
```

This dramatically reduces trading costs with minimal impact on returns.

### 2.3 Signal-Based Rebalancing vs Hold-Until-Flip

| Approach | Description | Used By | Pros | Cons |
|----------|-------------|---------|------|------|
| Continuous rebalancing | Adjust position daily based on forecast | Carver, Man AHL, AQR | Smooth, lower risk | Higher turnover |
| Hold-until-flip | Hold full position until signal reverses | Basic trend systems | Simple, low turnover | Binary risk, whipsaws |
| Threshold rebalancing | Rebalance when position deviates >X% | Most institutional | Balance of above | Needs calibration |

**Industry consensus: Continuous with buffering is superior.** Carver demonstrated this conclusively.

### 2.4 Pyramiding Rules for Trend Following (If Used)

For traders who DO want discrete pyramiding:

1. **Decreasing size**: Each layer smaller than previous (e.g., 4-3-2-1 lots)
2. **ATR-based spacing**: Add every 1-2 ATR in profit direction
3. **Max layers**: 3-4 typically
4. **Move stops**: Trail stops to breakeven on prior entries
5. **Total risk cap**: Never exceed 2% total portfolio risk across all layers
6. **Only in trends**: Never pyramid in ranging markets

---

## 3. Portfolio-Level Sizing: Multiple Strategies on Same Instrument

### 3.1 The Three Options

#### Option A: Capital Splitting
```
Total capital: $1M
Strategy 1 weight: 60% → $600K capital, sizes independently
Strategy 2 weight: 40% → $400K capital, sizes independently
Final position = Sum of individual positions
```

**Pros:** Simple, strategies are independent
**Cons:** Can over-leverage when strategies agree; doesn't account for correlation between strategies

#### Option B: Signal Blending (Carver's Approach) ★ RECOMMENDED
```
Combined_Forecast = Σ(forecast_i × weight_i)
  where Σ(weight_i) = 1.0

Then size ONE combined position using the blended forecast.
```

**This is what Robert Carver and pysystemtrade use.** This is also conceptually what most large CTAs do.

**Example (from Carver's actual system):**
```python
# Forecast weights for a single instrument (e.g., EDOLLAR):
weights = {
    'assettrend32': 0.048,
    'assettrend64': 0.048,
    'breakout160':  0.048,
    'carry125':     0.079,
    'momentum32':   0.048,
    'normmom64':    0.048,
    'relcarry':     0.127,
    # ... etc, all weights sum to 1.0
}

combined_forecast = sum(forecast[rule] * weights[rule] for rule in rules)
# Then cap at [-20, +20]
# Then calculate ONE position
```

#### Option C: Risk Budget Allocation
```
Each strategy gets a risk budget (e.g., 1% annual vol each)
Positions are independent but constrained by total risk budget
Portfolio risk = √(Σ w²σ² + 2Σ wᵢwⱼσᵢⱼ)
```

### 3.2 Which Approach Do Major CTAs Use?

**Signal blending (Option B) is the industry standard for multiple models on the same instrument.**

| Firm | Approach | Details |
|------|----------|---------|
| Robert Carver / pysystemtrade | Signal blending | Weighted average of ~40 forecast variants |
| Man AHL | Signal blending | Multiple model speeds blended before sizing |
| AQR | Signal blending + risk targeting | Blend across speeds, then vol-target the result |
| Winton | Signal blending | Multi-model forecast combination |
| Campbell | Signal blending | Diversified across short/medium/long-term signals |
| Kevin Davey | Position summing | Sums positions from independent strategies |

**Why signal blending wins:**
1. Avoids double-counting risk when strategies agree
2. Natural diversification: if one says +20 and another says -10, net = +10 (moderate)
3. Single position is cheaper to execute than multiple overlapping positions
4. Forecast Diversification Multiplier (FDM) accounts for imperfect correlation between forecasts

### 3.3 Carver's Forecast Combination Framework

```
Step 1: Group similar trading rules (e.g., all trend rules together)
Step 2: Equal weight within groups (handcrafting)
Step 3: Weight groups (e.g., trend 60%, carry 18%, other 22%)
Step 4: Apply Forecast Diversification Multiplier (FDM)
Step 5: Cap combined forecast at [-20, +20]
```

**Carver's actual strategy weights:**
```
Trend group:     60%  (assettrend, momentum, normmom, breakout, relmomentum)
Carry group:     18%  (carry at various speeds)
Other:           22%  (relcarry, skew, mean reversion, acceleration)
```

**FDM (Forecast Diversification Multiplier):**
- Similar to IDM but for trading rules
- Accounts for less-than-perfect correlation between forecasts
- Typically 1.0 - 2.5 depending on number and diversity of rules
- Capped at reasonable level to avoid over-leveraging

### 3.4 CFA Institute Research: Signal Blending vs Portfolio Blending

From Patel (2018), "Comparing Portfolio Blending and Signal Blending" (Financial Analysts Journal):
- **Signal blending** produces more concentrated portfolios with higher expected returns
- **Portfolio blending** produces more diversified portfolios
- For multifactor strategies, signal blending slightly outperforms
- **For trend following specifically, signal blending is dominant** because you want one net position per instrument

---

## 4. Capital Allocation Across Strategies (Instrument Weights)

### 4.1 Equal Risk Contribution (Risk Parity)

```
Risk_contribution_i = w_i × ∂σ_portfolio/∂w_i = w_i × (Σ_i × w) / σ_portfolio

Target: RC_1 = RC_2 = ... = RC_n
```

**Huatai Securities approach:**
- Use inverse volatility as weights
- Rebalance every 20 trading days
- This is a simplified risk parity (ignores correlations)

### 4.2 Carver's Handcrafting Method

**Step-by-step:**
1. Group instruments by correlation (e.g., "Bond & STIR", "Equity", "Ags")
2. Equal weight within groups
3. Weight groups by expected diversification benefit
4. Apply Instrument Diversification Multiplier (IDM)

**Carver's actual instrument group weights:**
```
Ags:              15%
Bond & STIR:      19%
Equity:           22%
FX:               13%
Metals & Crypto:  13%
OilGas:           13%
Vol:               5%
```

**IDM (Instrument Diversification Multiplier):**
- Accounts for imperfect correlation between instruments
- Calculated from subsystem return correlations (≈70% of underlying instrument correlations)
- Typically 1.0 - 2.5, capped at 2.5
- Example: 3 instruments, avg correlation 0.25 → IDM ≈ 1.41

### 4.3 Strategy-Level Vol Targeting

Each strategy targets the same volatility:
```
For instrument i with strategy j:
  Position_ij = (Forecast_ij / 10) × Vol_Scalar_i × Instrument_Weight_i × IDM × FDM
```

All subsystems are already volatility-standardized to the same expected standard deviation of returns. This enables direct weight comparison.

### 4.4 Preventing Over-Leveraging When Multiple Strategies Agree

**This is the critical question. Solutions:**

1. **Signal blending naturally handles this:** Combined forecast is capped at [-20, +20], so even if all 40 rules say +20, the combined forecast maxes at +20 (2x average position)

2. **Risk overlay (Carver's 3-component system):**
   - **Max expected risk:** `risk_multiplier = min(1, 2 × target_risk / current_expected_risk)` — reduces positions when portfolio risk > 2× target
   - **Correlation risk:** Replace correlation matrix with worst case (all correlations = 1), use absolute weights. `risk_multiplier = min(1, 4 × target_risk / worst_case_risk)`
   - **Std dev risk:** Use 99th percentile of historical vol. `risk_multiplier = min(1, 6 × target_risk / 99vol_risk)`
   - Take the minimum (most conservative) of all three

3. **Man AHL maximum diversification approach:** Allocation via maximum diversification, incorporating volatility, correlation, and liquidity by market

4. **Margin check:** Total margin usage should not exceed 30-50% of capital (absolute guardrail)

---

## 5. Position Limits and Guardrails

### 5.1 Industry-Standard Limits

| Limit Type | Typical Value | Source |
|------------|--------------|-------|
| Max position as % of daily volume | 1-5% of 20-day avg volume | Industry standard |
| Max single instrument risk | 2-5% of portfolio risk | Most CTAs |
| Max sector concentration | 20-30% of total risk | Institutional |
| Max margin utilization | 30-50% of capital | Prudent risk management |
| Max leverage (notional/capital) | 5-15x | Varies by strategy |
| Forecast cap | ±20 (2x average) | Carver |
| IDM cap | 2.5 | Carver |
| Position inertia buffer | 10% of target | Carver |

### 5.2 Carver's Risk Overlay Parameters (Default)

```yaml
risk_overlay:
  max_risk_fraction_normal_risk: 2.0      # Cut when expected risk > 2× target
  max_risk_fraction_correlation_risk: 4.0  # Cut when worst-case corr risk > 4× target
  max_risk_fraction_stdev_risk: 6.0       # Cut when 99th pct vol risk > 6× target
```

### 5.3 Chinese Commodity Futures Position Limits (中国期货交易所持仓限额)

**SHFE (上海期货交易所) - Key Limits (2025 rules):**

| 品种 | 单日最大开仓限额 | 特殊说明 |
|------|-----------------|---------|
| 黄金 (au) | 2,800手 | 含夜盘 |
| 铜 (cu) | 2,000手 | 含套保 |
| 螺纹钢 (rb) | 32,000手 | 仅主力合约 |
| 原油 (sc) | 320手 | |

**DCE (大连商品交易所):**

| 品种 | 单日最大开仓限额 | 特殊说明 |
|------|-----------------|---------|
| 焦炭 (j) | 50手 | 含跨期头寸 |
| 生猪 (lh) | 1,000手 | |
| 铁矿石 (i) | 2,000手 | 含境外交易者 |
| 豆粕 (m) | 20,000手 | 含期权 |

**CZCE (郑州商品交易所):**

| 品种 | 单日最大开仓限额 | 特殊说明 |
|------|-----------------|---------|
| 动力煤 (ZC) | 20手 | 极低 |
| 尿素 (UR) | 2,000手 | |
| PTA | 30,000手 | |
| 白糖 (SR) | 10,000手 | |

**Key Chinese exchange rules:**
1. **持仓限额制度**: Position limits vary by contract month (一般月份 vs 交割月前一月 vs 交割月)
2. **Percentage-based limits**: For liquid contracts (e.g., volume > 100K lots), limit = 10% of open interest
3. **Absolute limits**: Smaller for delivery months
4. **动态调整**: Exchanges can tighten limits during high volatility (e.g., 动力煤 price swing > 5%)
5. **合并计算**: Same client positions across different brokers are consolidated
6. **套保豁免**: Hedgers can apply for exemption above limits

**Campbell & Company finding on capacity:**
- Capacity factor has negative Sharpe ratio (-0.30) from 2001-2015
- Re-allocating risk due to position limits costs ~0.94% per year on average
- However, some CTA managers show positive exposure to capacity factor
- Position limits cause performance to deviate from equal-risk benchmark

### 5.4 Practical Max Position Size Formula

```
Max_contracts = min(
    Vol_target_contracts,                    # From vol targeting
    Daily_volume × Volume_fraction,          # Liquidity constraint (1-5%)
    Exchange_position_limit,                 # Regulatory limit
    (Max_margin_pct × Capital) / Initial_margin,  # Margin constraint
    Absolute_max_contracts                   # Hard stop
)
```

---

## 6. Rebalancing Frequency

### 6.1 How Often Do CTAs Rebalance?

| Rebalancing Type | Frequency | Used By |
|-----------------|-----------|---------|
| Continuous (daily signals) | Daily recalculation, trade if needed | Man AHL, AQR, Carver |
| With position inertia | Daily calc, trade only if >10% deviation | Carver (recommended) |
| Periodic + threshold | Weekly/monthly, trade if >threshold | Some smaller CTAs |
| Monthly risk parity | Every 20 trading days | Huatai Securities CTA model |
| Fixed calendar | Monthly | Simplest approaches |

### 6.2 Carver's Rebalancing Approach

**Positions are recalculated daily but trades are filtered by position inertia:**

```
if abs(current_position - target_position) > 0.10 × abs(target_position):
    TRADE to target
else:
    HOLD current position
```

**Additionally:**
- If vol target < 15%: no need to adjust daily (weekly is fine)
- Correlations: recalculated using 120-day span (updated less frequently)
- Std devs: recalculated using 30-day EMA span
- Risk overlay covariance matrix: monthly recalculation (but using daily position weights)
- Instrument weights: updated monthly or quarterly

### 6.3 Cost of Rebalancing vs Tracking Error

From research:
- **Quarterly rebalancing** captures most of the benefit while minimizing costs
- **Daily with buffering** is optimal for CTA (because signals change)
- **Threshold-based** (band-based) consistently outperforms calendar-based
- **Optimal threshold**: ~5-10% deviation from target for large portfolios

**Carver's finding:** Position inertia of 10% reduces trading costs significantly with minimal performance impact (SR difference <0.02).

---

## 7. Summary: Practical Recommendations for QBase

### Position Sizing Pipeline

```
┌──────────────────┐
│  Individual       │
│  Strategy         │    For each (instrument, strategy):
│  Forecasts        │    forecast ∈ [-20, +20]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Signal Blending  │    combined_forecast = Σ(forecast_i × weight_i)
│  (per instrument) │    Cap at [-20, +20]
│                   │    Apply FDM
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Position Sizing  │    position = (combined_forecast/10)
│  (vol targeting)  │              × vol_scalar
│                   │              × instrument_weight × IDM
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Position Inertia │    if deviation < 10%: hold
│  (buffering)      │    else: trade to target
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Risk Overlay     │    Check: expected risk, correlation risk, vol risk
│  (guardrails)     │    Multiply all positions by min(multiplier_1, _2, _3)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Position Limits  │    Apply: exchange limits, volume limits, margin limits
│  (hard stops)     │
└──────────────────┘
```

### Key Parameters

```yaml
# Volatility Targeting
annual_vol_target: 0.25          # 25% (half-Kelly for SR~0.75)
vol_lookback_days: 36            # EMA span for daily vol estimation
annualization_factor: 16         # √256

# Forecasting
average_absolute_forecast: 10
forecast_cap: 20                 # ±20
forecast_floor: -20

# Diversification
max_IDM: 2.5                    # Instrument Diversification Multiplier cap
max_FDM: 2.5                    # Forecast Diversification Multiplier cap

# Position Management
position_inertia_threshold: 0.10  # 10% buffer
rebalance_frequency: daily        # With inertia filtering

# Risk Overlay
max_normal_risk_multiple: 2.0
max_correlation_risk_multiple: 4.0
max_stdev_risk_multiple: 6.0

# Position Limits
max_position_pct_daily_volume: 0.05  # 5% of daily volume
max_margin_utilization: 0.50         # 50% of capital
max_single_instrument_risk_pct: 0.05 # 5% of total risk

# Chinese Exchange Specific
check_exchange_position_limits: true
position_limit_buffer: 0.80          # Use only 80% of exchange limit
```

---

## 8. Key References

1. **Robert Carver, "Systematic Trading" (2015)** - Foundation of vol targeting, forecast sizing
2. **Robert Carver, "Leveraged Trading" (2019)** - Simplified version for retail traders
3. **Robert Carver, "Advanced Futures Trading Strategies" (2023)** - Dynamic optimization, 250+ markets
4. **Robert Carver blog** (qoppac.blogspot.com) - Risk overlay, forecast weights, vol targeting
5. **pysystemtrade** (github.com/pst-group/pysystemtrade) - Full open-source implementation
6. **Harvey et al. "The Impact of Volatility Targeting" (2018)** - JPM, vol targeting improves Sharpe
7. **Kaminski, "Quantifying CTA Risk Management" (2015)** - Campbell & Co, risk factor framework
8. **AQR, "Demystifying Managed Futures" (2013)** - TSMOM replicates CTA returns
9. **AQR, "Understanding Risk Parity" (2010)** - Equal risk contribution framework
10. **Man AHL, "Optimal Market Mix for Trend Follower" (2026)** - Max Sharpe vs Max Crisis Sharpe
11. **华泰金工林晓明团队, "基于风险平价的CTA组合策略" (2020)** - 风险平价CTA, 夏普2.045
12. **中国证监会, "期货市场持仓管理暂行规定" (2023)** - 持仓限额法规
13. **Baltas & Kosowski, "Trend-Following, Risk-Parity" (2015)** - Academic framework
14. **Patel, "Signal Blending vs Portfolio Blending" (2018)** - CFA Institute, FAJ
