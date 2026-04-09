# CTA System Architecture: Deep Research Report

## Based on pysystemtrade (Robert Carver) and industry practices

---

## Table of Contents

- [A. The Exact Processing Pipeline](#a-the-exact-processing-pipeline)
- [B. Exact Formulas for Key Calculations](#b-exact-formulas-for-key-calculations)
- [C. How Weights Are Estimated (Handcrafting)](#c-how-weights-are-estimated-handcrafting)
- [D. Multi-Instrument Handling](#d-multi-instrument-handling)
- [E. Data Management](#e-data-management)
- [F. Parameter Estimation Methodology](#f-parameter-estimation-methodology)
- [G. Risk Management Layers](#g-risk-management-layers)
- [H. Recommended Project Structure](#h-recommended-project-structure)

---

## A. The Exact Processing Pipeline

### Architecture Overview

pysystemtrade uses a **stage-based DAG (Directed Acyclic Graph)** architecture. The system is composed of `SystemStage` objects that are assembled into a `System` object. Each stage has:

- **Inputs**: data from previous stages (accessed via `self.parent.<stage_name>.<method>()`)
- **Outputs**: cached results decorated with `@output()`, `@diagnostic()`, or `@input`
- **Caching**: all intermediate results are cached in a `systemCache` to avoid recomputation

```
System(
    stage_list=[RawData, Rules, ForecastScaleCap, ForecastCombine, PositionSizing, Portfolios],
    data=csvFuturesSimData(),
    config=Config("config.yaml")
)
```

### The 7 Pipeline Stages (in order)

```
data (SimData)
  |
  v
Stage 1: RawData           -- prices, returns, volatility, carry data
  |
  v
Stage 2: Rules             -- raw forecasts from trading rules
  |
  v
Stage 3: ForecastScaleCap  -- scale raw forecasts, then cap at +/-20
  |
  v
Stage 4: ForecastCombine   -- weighted combination of forecasts + FDM
  |
  v
Stage 5: PositionSizing    -- convert combined forecast to subsystem position
  |
  v
Stage 6: Portfolios        -- apply instrument weights, IDM, risk overlay, buffers
  |
  v
Final Output: actual_position per instrument
```

### Stage 1: RawData

**Name**: `rawdata`

**Purpose**: Preprocessing layer. Computes daily prices, returns, volatility, and carry data. These are cached so downstream stages don't recompute them.

**Key outputs**:
| Method | Output | Description |
|--------|--------|-------------|
| `get_daily_prices(instrument)` | pd.Series | Back-adjusted continuous daily prices |
| `daily_returns(instrument)` | pd.Series | `price.diff()` |
| `daily_returns_volatility(instrument)` | pd.Series | EW std of returns (default 35-day span) |
| `annualised_returns_volatility(instrument)` | pd.Series | `daily_vol * sqrt(256)` |
| `get_daily_percentage_volatility(instrument)` | pd.Series | `100 * vol / denominator_price` |
| `get_daily_vol_normalised_returns(instrument)` | pd.Series | `daily_return / vol_yesterday` |
| `get_cumulative_daily_vol_normalised_returns(instrument)` | pd.Series | cumsum of above (used for breakout/rel_mom) |
| `raw_carry(instrument)` | pd.Series | annualised_roll / annualised_vol |
| `smoothed_carry(instrument)` | pd.Series | EWM(90) of raw_carry |
| `daily_denominator_price(instrument)` | pd.Series | price used for % vol calculations |

**Volatility calculation (robust_vol_calc)**:
```python
def robust_vol_calc(daily_returns, days=35, min_periods=10,
                    vol_abs_min=1e-10, vol_floor=True,
                    floor_min_quant=0.05, floor_min_periods=100, floor_days=500):
    vol = daily_returns.ewm(span=days, min_periods=min_periods).std()
    vol[vol < vol_abs_min] = vol_abs_min
    if vol_floor:
        vol_min = vol.rolling(window=floor_days, min_periods=floor_min_periods).quantile(0.05)
        vol_min.iloc[0] = 0.0
        vol_min = vol_min.ffill()
        vol = max(vol, vol_min)
    return vol
```

Key insight: The vol floor prevents vol from dropping below its 5th percentile over the last 500 days. This prevents the system from taking excessively large positions during unusually calm periods.

### Stage 2: Rules (Trading Rules)

**Name**: `rules`

**Purpose**: Apply trading rule functions to produce raw (unscaled) forecasts.

**Architecture**: Each trading rule is a `TradingRule` object containing:
- `function`: the callable that computes the forecast
- `data`: list of string paths to data methods (e.g., `"rawdata.get_daily_prices"`)
- `other_args`: dict of keyword arguments (e.g., `{"Lfast": 16, "Lslow": 64}`)

**Key output**: `get_raw_forecast(instrument_code, rule_variation_name)` -> pd.Series

Rules are defined in config YAML:
```yaml
trading_rules:
  ewmac16_64:
    function: systems.provided.rules.ewmac.ewmac
    data:
      - rawdata.get_daily_prices
      - rawdata.daily_returns_volatility
    other_args:
      Lfast: 16
      Lslow: 64
```

### Stage 3: ForecastScaleCap

**Name**: `forecastScaleCap`

**Purpose**: Scale raw forecasts so they have an average absolute value of 10, then cap at +/- 20.

**Pipeline**:
```
raw_forecast
    * forecast_scalar
    = scaled_forecast
    .clip(-20, +20)
    = capped_forecast
```

**Two modes**:
1. **Fixed**: forecast scalar from config (e.g., `forecast_scalars: {ewmac16_64: 5.3}`)
2. **Estimated**: computed from data using expanding window

### Stage 4: ForecastCombine

**Name**: `combForecast`

**Purpose**: Weighted combination of capped forecasts, multiplied by Forecast Diversification Multiplier (FDM).

**Pipeline**:
```
For each rule r:
    weighted_forecast_r = capped_forecast_r * weight_r

# Weights are renormalized when some forecasts are missing (NaN)
combined_forecast = sum(weighted_forecasts) * FDM

# Then cap again at +/- 20
final_combined_forecast = clip(combined_forecast, -20, +20)
```

**Two modes for weights**:
1. **Fixed**: from config `forecast_weights`
2. **Estimated**: handcrafting or other optimisation methods

### Stage 5: PositionSizing

**Name**: `positionSize`

**Purpose**: Convert combined forecast into a subsystem position (number of contracts), assuming we trade our entire capital on this one instrument.

**The core formula**:
```
subsystem_position = vol_scalar * combined_forecast / avg_abs_forecast

where:
    vol_scalar = daily_cash_vol_target / instrument_value_vol
    daily_cash_vol_target = (capital * percentage_vol_target / 100) / sqrt(256)
    instrument_value_vol = instrument_currency_vol * fx_rate
    instrument_currency_vol = block_value * daily_percentage_vol
    block_value = denominator_price * value_of_price_move * 0.01
    avg_abs_forecast = 10.0 (by convention)
```

**Key output**: `get_subsystem_position(instrument_code)` -> pd.Series

Also computes subsystem-level buffers.

### Stage 6: Portfolios

**Name**: `portfolio`

**Purpose**: Apply instrument weights, IDM, risk overlay, and position buffers.

**Pipeline**:
```
notional_position_without_idm = subsystem_position * instrument_weight
notional_position_before_risk = notional_position_without_idm * IDM
notional_position = notional_position_before_risk * risk_scalar  (if risk overlay enabled)
actual_position = notional_position * capital_multiplier

buffers = calculate_buffers(position, vol_scalar, idm, instr_weights)
```

**Key outputs**:
- `get_notional_position(instrument_code)` -> pd.Series
- `get_actual_position(instrument_code)` -> pd.Series (with capital scaling)
- `get_buffers_for_position(instrument_code)` -> pd.DataFrame (top_pos, bot_pos)

### Data Flow Summary

```
Per instrument, per rule:
    price -> raw_forecast -> scaled_forecast -> capped_forecast

Per instrument, across rules:
    [capped_forecasts] * weights -> combined_forecast * FDM -> capped_combined

Per instrument:
    capped_combined * vol_scalar / 10 -> subsystem_position
    subsystem_position * instrument_weight * IDM * risk_scalar -> notional_position

Portfolio level:
    risk_overlay -> risk_scalar (applied to all positions)
    buffers -> position bands [top_pos, bot_pos]
```

---

## B. Exact Formulas for Key Calculations

### B.1 EWMAC (Exponentially Weighted Moving Average Crossover)

The most important trend-following signal. From `systems/provided/rules/ewmac.py`:

```python
def ewmac(price, vol, Lfast, Lslow):
    fast_ewma = price.ewm(span=Lfast, min_periods=1).mean()
    slow_ewma = price.ewm(span=Lslow, min_periods=1).mean()
    raw_ewmac = fast_ewma - slow_ewma
    return raw_ewmac / vol.ffill()
```

**Formula**:
```
EWMAC = (EMA_fast(price) - EMA_slow(price)) / daily_price_vol
```

**Standard variations** (Carver uses Lslow = 4 * Lfast):
| Name | Lfast | Lslow |
|------|-------|-------|
| ewmac2 | 2 | 8 |
| ewmac4 | 4 | 16 |
| ewmac8 | 8 | 32 |
| ewmac16 | 16 | 64 |
| ewmac32 | 32 | 128 |
| ewmac64 | 64 | 256 |

**Important notes**:
- `price` is the back-adjusted continuous price series
- `vol` is `robust_vol_calc(price.diff(), days=35)` - NOT percentage vol
- Output is in "number of standard deviations" space
- After scaling by forecast scalar, output centers around absolute value of 10
- The vol-division makes it comparable across instruments and time periods

### B.2 Breakout

From `systems/provided/rules/breakout.py`:

```python
def breakout(price, lookback=10, smooth=None):
    if smooth is None:
        smooth = max(int(lookback / 4.0), 1)

    roll_max = price.rolling(lookback, min_periods=ceil(lookback/2)).max()
    roll_min = price.rolling(lookback, min_periods=ceil(lookback/2)).min()
    roll_mean = (roll_max + roll_min) / 2.0

    output = 40.0 * ((price - roll_mean) / (roll_max - roll_min))
    smoothed_output = output.ewm(span=smooth, min_periods=ceil(smooth/2)).mean()
    return smoothed_output
```

**Formula**:
```
raw_breakout = 40 * (price - midpoint) / (range)

where:
    midpoint = (rolling_max + rolling_min) / 2
    range = rolling_max - rolling_min
```

**Notes**:
- The `40.0` multiplier gives natural scaling: when price is at the top of the range, output is +20; at bottom, -20.
- This signal is ALREADY naturally scaled to [-20, +20] range.
- Standard variations: breakout10, breakout20, breakout40, breakout80, breakout160, breakout320
- Breakout uses the vol-normalised cumulative price, NOT the raw price. The `price` input here is `rawdata.get_cumulative_daily_vol_normalised_returns()`.

### B.3 Carry

From `systems/provided/rules/carry.py` and `systems/rawdata.py`:

```python
# In rawdata:
def raw_carry(instrument_code):
    daily_ann_roll = daily_annualised_roll(instrument_code)
    vol = daily_returns_volatility(instrument_code)
    ann_stdev = vol * sqrt(256)
    raw_carry = daily_ann_roll / ann_stdev
    return raw_carry

def annualised_roll(instrument_code):
    rawrollvalues = raw_futures_roll(instrument_code)   # carry_price - price
    rolldiffs = roll_differentials(instrument_code)     # fraction of year between contracts
    annroll = rawrollvalues / rolldiffs
    return annroll

# The trading rule itself:
def carry(raw_carry, smooth_days=90):
    smooth_carry = raw_carry.ewm(smooth_days).mean()
    return smooth_carry
```

**Formula**:
```
raw_futures_roll = carry_contract_price - price_contract_price
annualised_roll = raw_futures_roll / (years_between_contracts)
raw_carry = annualised_roll / (daily_vol * sqrt(256))
carry_forecast = EWM(raw_carry, span=90)
```

**Carry data requires**:
- `PRICE`: price of the contract being traded
- `CARRY`: price of the carry contract (typically the next contract out)
- `CARRY_CONTRACT` and `PRICE_CONTRACT`: contract identifiers for computing time between them

**Standard variations**: carry10, carry30, carry60, carry125 (smooth_days parameter)

### B.4 Relative Momentum

From `systems/provided/rules/rel_mom.py`:

```python
def relative_momentum(normalised_price_this_instrument,
                       normalised_price_for_asset_class,
                       horizon=250, ewma_span=None):
    if ewma_span is None:
        ewma_span = int(horizon / 4.0)

    outperformance = normalised_price_this_instrument - normalised_price_for_asset_class
    average_outperformance = (outperformance - outperformance.shift(horizon)) / horizon
    forecast = average_outperformance.ewm(span=ewma_span).mean()
    return forecast
```

**Formula**:
```
outperformance = vol_normalised_cum_return_instrument - vol_normalised_cum_return_asset_class
avg_outperformance = (outperformance - outperformance[t-horizon]) / horizon
forecast = EWM(avg_outperformance, span=horizon/4)
```

### B.5 Cross-Sectional Mean Reversion

From `systems/provided/rules/cs_mr.py`:

```python
def cross_sectional_mean_reversion(normalised_price_this, normalised_price_class,
                                     horizon=250, ewma_span=None):
    if ewma_span is None:
        ewma_span = int(horizon / 4.0)

    outperformance = normalised_price_this - normalised_price_class
    relative_return = outperformance.diff()
    outperformance_over_horizon = relative_return.rolling(horizon).mean()
    forecast = -outperformance_over_horizon.ewm(span=ewma_span).mean()
    return forecast
```

**Key**: Note the **negative sign** -- this is mean reversion, so we fade the outperformance.

### B.6 Acceleration

From `systems/provided/rules/accel.py`:

```python
def accel(price, vol, Lfast=4):
    Lslow = Lfast * 4
    ewmac_signal = ewmac(price, vol, Lfast, Lslow)
    accel = ewmac_signal - ewmac_signal.shift(Lfast)
    return accel
```

**Formula**: `acceleration = EWMAC(t) - EWMAC(t - Lfast)` -- rate of change of trend.

### B.7 Mean Reversion Wings

From `systems/provided/rules/mr_wings.py`:

```python
def mr_wings(price, vol, Lfast=4):
    Lslow = Lfast * 4
    ewmac_signal = ewmac(price, vol, Lfast, Lslow)
    ewmac_std = ewmac_signal.rolling(5000, min_periods=3).std()
    ewmac_signal[ewmac_signal.abs() < ewmac_std * 3] = 0.0
    mr_signal = -ewmac_signal
    return mr_signal
```

**Formula**: Fade extreme (>3 sigma) EWMAC signals. Zero otherwise.

### B.8 Forecast Scalar

The forecast scalar ensures the average absolute value of a forecast equals 10 (the "average absolute forecast").

From `sysquant/estimators/forecast_scalar.py`:

```python
def forecast_scalar(cs_forecasts, target_abs_forecast=10.0,
                    window=250000, min_periods=500, backfill=True):
    copy_cs_forecasts = cs_forecasts.copy()
    copy_cs_forecasts[copy_cs_forecasts == 0.0] = np.nan

    # Cross-sectional median of absolute values first
    if copy_cs_forecasts.shape[1] == 1:
        x = copy_cs_forecasts.abs().iloc[:, 0]
    else:
        x = copy_cs_forecasts.ffill().abs().median(axis=1)

    # Then time-series rolling mean
    avg_abs_value = x.rolling(window=window, min_periods=min_periods).mean()
    scaling_factor = target_abs_forecast / avg_abs_value

    if backfill:
        scaling_factor = scaling_factor.bfill()

    return scaling_factor
```

**Key methodology**:
1. Pool forecasts across instruments (cross-sectional median)
2. Take rolling expanding window mean of abs values
3. Scalar = 10 / avg_abs_value
4. Backfill the first estimate to avoid NaN period

**Typical forecast scalar values** (from Carver's book):
| EWMAC variation | Approximate scalar |
|-----------------|-------------------|
| ewmac2_8 | 12.1 |
| ewmac4_16 | 8.53 |
| ewmac8_32 | 5.95 |
| ewmac16_64 | 4.10 |
| ewmac32_128 | 2.79 |
| ewmac64_256 | 1.91 |

### B.9 Forecast Diversification Multiplier (FDM)

```python
def diversification_mult_single_period(corrmatrix, weights, dm_max=2.5):
    risk = weights.portfolio_stdev(corrmatrix)  # sqrt(W * H * W^T)
    if risk < 1e-7:
        return 1.0
    dm = min(1.0 / risk, dm_max)
    return dm
```

**Formula**:
```
FDM = 1 / sqrt(W * H * W^T)

where:
    W = vector of forecast weights
    H = correlation matrix of forecasts
    capped at dm_max = 2.5
```

The FDM is computed over time using estimated correlations, then smoothed with EWM(span=125).

**Typical FDM values**: 1.0 to 2.5 depending on number of rules and their correlations.

### B.10 Instrument Diversification Multiplier (IDM)

Same formula as FDM but at the portfolio level:

```
IDM = 1 / sqrt(W_instr * H_instr * W_instr^T)

where:
    W_instr = vector of instrument weights
    H_instr = correlation matrix of instrument returns
    capped at dm_max = 2.5
```

**Typical IDM values**:
| Number of instruments | Approximate IDM |
|----------------------|-----------------|
| 1 | 1.0 |
| 2-3 | 1.2-1.5 |
| 4-6 | 1.5-1.8 |
| 7-15 | 1.8-2.2 |
| 15+ | 2.0-2.5 |

### B.11 Position Sizing (Complete Formula)

```
N = (Capital * VolTarget% / 100) / (sqrt(256) * BlockValue * DailyPriceVol% * FX) * (Forecast / 10) * InstrWeight * IDM * RiskScalar

Simplified step by step:
    daily_cash_vol_target = Capital * VolTarget% / 100 / sqrt(256)
    instrument_currency_vol = denominator_price * point_value * 0.01 * daily_%_vol
    instrument_value_vol = instrument_currency_vol * fx_rate
    vol_scalar = daily_cash_vol_target / instrument_value_vol
    subsystem_position = vol_scalar * forecast / 10
    notional_position = subsystem_position * instrument_weight * IDM * risk_scalar
```

**Worked example** (from Carver's blog):
```
Capital = $500,000
Vol target = 25% annual
Daily cash vol target = 500000 * 0.25 / 16 = $7,812.50

For Crude Oil:
    Price = $60, Point value = $1,000, Daily vol = 2%
    Block value = $60 * $1000 * 0.01 = $600
    Instrument currency vol = $600 * 0.02 = $12/day per contract? No...
    Actually: instrument_currency_vol = block_value * daily_percentage_vol
    block_value = price * point_value * 0.01 = 60 * 1000 * 0.01 = 600
    daily_pct_vol = daily_returns_vol / price = daily_vol_in_points / price
    If daily vol in points = $1.20, daily_pct_vol = 1.20/60 = 0.02 = 2%
    instrument_currency_vol = 600 * 0.02 = $12? That's too low...

Correct interpretation:
    daily_returns_vol = robust_vol_calc(price.diff()) = e.g. 1.20 points
    annualised_returns_vol = 1.20 * 16 = 19.2 points
    block_value = price * value_of_block_price_move * 0.01
    For Crude: value_of_block_price_move = 100000 (i.e. 1 point move = $1000,
               but stored as 100000 for the "0.01" math)
    Actually the code does: block_value = underlying_price * value_of_price_move * 0.01

    instrument_currency_vol = block_value * daily_percentage_vol
    daily_percentage_vol = 100 * daily_vol / denom_price

    So: instrument_currency_vol = (price * multiplier * 0.01) * (100 * daily_vol / price)
                                = multiplier * daily_vol
                                = 1000 * 1.20 = $1,200/day per contract

    vol_scalar = $7,812.50 / $1,200 = 6.51 contracts

    If forecast = +10 (neutral), position = 6.51 * 10/10 = 6.51 contracts
    If forecast = +20 (max), position = 6.51 * 20/10 = 13.02 contracts
    If forecast = -5, position = 6.51 * -5/10 = -3.26 contracts
```

---

## C. How Weights Are Estimated (Handcrafting Methodology)

Carver's handcrafting method is designed to produce robust weights without overfitting. It is implemented in `sysquant/optimisation/full_handcrafting.py`.

### C.1 The Core Problem

Traditional mean-variance optimisation is fragile because:
- Expected returns are extremely hard to estimate
- Correlation matrices are noisy
- Small changes in inputs cause large changes in weights

### C.2 Handcrafting Steps

**Step 1: Hierarchical Clustering**

Group assets/rules into clusters of maximum size 3 using hierarchical clustering on the correlation matrix.

```
MAX_CLUSTER_SIZE = 3
```

**Step 2: Within-Cluster Weight Calculation with Uncertainty**

For each cluster of 2-3 assets, compute weights considering correlation uncertainty:

```python
def optimised_weights_given_correlation_uncertainty(corr_matrix, data_points, p_step=0.25):
    dist_points = np.arange(p_step, stop=(1-p_step)+1e-6, step=p_step)
    list_of_weights = []

    for conf1 in dist_points:
        for conf2 in dist_points:
            for conf3 in dist_points:
                # For each combination of confidence intervals on correlations:
                # 1. Get correlation values at those confidence points
                # 2. Build correlation matrix
                # 3. Optimise weights for that matrix
                weights = optimise_for_corr_matrix(adjusted_corr_matrix)
                list_of_weights.append(weights)

    # Average across all scenarios
    average_weights = np.nanmean(array_of_weights, axis=0)
    return average_weights
```

**Step 3: Fisher Transform for Correlation Uncertainty**

Correlation uncertainty is modeled using the Fisher transform:

```python
def fisher_transform(corr_value):
    return 0.5 * np.log((1 + corr_value) / (1 - corr_value))  # arctanh

def fisher_stdev(data_points):
    return 1 / ((data_points - 3) ** 0.5)

# Confidence interval in Fisher space:
# fisher_corr +/- z * fisher_stdev * FUDGE_FACTOR (4.0)
```

The **fudge factor of 4.0** is critical - it massively inflates uncertainty, pushing weights toward equal weighting. This is Carver's key anti-overfitting measure.

**Step 4: Minimum Weight Application**

```python
APPROX_MIN_WEIGHT_IN_CORR_WEIGHTS = 0.1
# Any weight below 10% is set to 10%, then renormalized
```

**Step 5: SR (Sharpe Ratio) Adjustment**

If assets have different expected Sharpe ratios, apply a bootstrap-based adjustment:

```python
def multiplier_from_relative_SR(relative_SR, avg_correlation, years_of_data):
    ratio = mini_bootstrap_ratio(relative_SR, avg_correlation, years_of_data)
    return ratio  # multiplier on weight, typically 0.5 to 1.5
```

This uses a mini-bootstrap to estimate the probability that the SR difference is real vs. noise.

**Step 6: Top-Down Assembly**

After computing weights within each cluster, weights are multiplied up through the hierarchy:

```
If cluster A has assets [X, Y, Z] with weights [0.4, 0.35, 0.25]
And cluster B has assets [P, Q] with weights [0.5, 0.5]
And top-level weights are [cluster_A: 0.6, cluster_B: 0.4]

Then: X=0.24, Y=0.21, Z=0.15, P=0.20, Q=0.20
```

### C.3 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| MAX_CLUSTER_SIZE | 3 | Maximum assets per cluster |
| APPROX_MIN_WEIGHT | 0.10 | Minimum weight before normalisation |
| FUDGE_FACTOR | 4.0 | Inflates correlation uncertainty |
| MAX_ROWS_FOR_CORR | 100 | Data points for correlation estimation |
| P_STEP | 0.25 | Step size for confidence interval grid |

### C.4 When to Use Handcrafting vs Fixed

- **Fixed weights**: If you have < 5 years of data, or few assets. Use equal weights.
- **Estimated (handcrafting)**: If you have > 10 years, many instruments/rules. Still very conservative.
- **Never use**: unconstrained mean-variance optimisation. It will overfit.

---

## D. Multi-Instrument Handling

### D.1 From Single to Multi-Instrument

The key insight: each instrument is first treated as a standalone subsystem (Stage 5), then combined at the portfolio level (Stage 6).

```
Per-instrument subsystem position = vol_scalar * forecast / 10
    (this is the position IF we traded only this instrument with all our capital)

Portfolio position = subsystem_position * instrument_weight * IDM
    (this scales down to account for capital allocation)
```

### D.2 Instrument Weight Estimation

Two approaches:

**Fixed (from config)**:
```yaml
instrument_weights:
  SP500: 0.10
  EUROSTX: 0.10
  US10: 0.10
  BUND: 0.10
  CRUDE_W: 0.10
  GOLD: 0.10
  ...
```

**Estimated**: Uses the same handcrafting methodology as forecast weights, but applied to subsystem returns (the P&L of each instrument's subsystem as if it were traded alone).

### D.3 Instrument Selection Criteria

From Carver's blog and books:

1. **Minimum volume**: Sufficient liquidity to trade without impact
2. **Minimum history**: At least 2-3 years, preferably 10+
3. **Cost filter**: Trading cost < 0.01 SR units per trade
4. **Correlation with existing instruments**: Avoid duplicates (correlation > 0.9)
5. **Minimum instrument weight**: At least 5% with IDM of 1.0, so the position is at least half a contract
6. **Practical**: Can you actually trade it? (margin, regulatory, etc.)

### D.4 Asset Class Diversification

Carver organises instruments by asset class for:
- Relative carry calculations (median carry within asset class)
- Cross-sectional mean reversion (vs. asset class average)
- Relative momentum (vs. asset class average)

Asset classes: Equity indices, Bonds, FX, Commodities (Metals, Energies, Ags), Rates (STIRs)

### D.5 IDM Calculation

```python
# Estimated: from correlation of instrument subsystem returns
corr_matrix = estimate_correlation_matrix(subsystem_returns)
IDM = 1 / sqrt(W * corr_matrix * W^T)
IDM = min(IDM, 2.5)

# Fixed: from config
instrument_div_multiplier: 2.0
```

---

## E. Data Management

### E.1 Data Format

pysystemtrade uses three types of futures data:

**1. Individual Contract Prices** (`futures_contract_prices/`):
```
Each file: {instrument}_{YYYYMMDD}.csv
Columns: OPEN, HIGH, LOW, FINAL, VOLUME
Index: datetime
```

**2. Multiple Prices** (`futures_multiple_prices/`):
One file per instrument, containing:
```
Columns: PRICE, CARRY, CARRY_CONTRACT, PRICE_CONTRACT, FORWARD, FORWARD_CONTRACT
- PRICE: price of the contract we're "trading"
- CARRY: price of the carry contract (for carry calculation)
- CARRY_CONTRACT: identifier of carry contract (e.g., 20240600)
- PRICE_CONTRACT: identifier of price contract
- FORWARD: price of the next contract (for roll detection)
- FORWARD_CONTRACT: identifier of forward contract
```

**3. Adjusted Prices** (`futures_adjusted_prices/`):
```
Single column of back-adjusted (Panama canal) continuous prices
Index: datetime
```

### E.2 Roll Handling

**Panama Canal Method** (back-adjustment):

When rolling from contract A to contract B:
```
adjustment = price_of_old_contract - price_of_new_contract  (at roll date)
all_historical_prices -= adjustment
```

This creates a continuous, gap-free price series. However:
- The absolute price level is meaningless (can be negative)
- Returns are preserved
- Used for: all trend-following signals, volatility calculation

**Roll date determination**:
- Carver uses a configurable roll calendar
- Typically rolls N business days before expiry
- The `FORWARD` column in multiple_prices tracks when the forward contract price is available
- Roll happens when volume shifts to the new contract, OR on a fixed schedule

### E.3 What You Actually Need

For backtesting, per instrument:
1. **Back-adjusted daily prices** (for signals and vol)
2. **Multiple prices** (for carry calculation: PRICE and CARRY columns)
3. **FX rates** (if trading non-base-currency instruments)
4. **Instrument config**: point_value, asset_class, currency, cost data

### E.4 Data Storage in Production

pysystemtrade supports:
- **CSV files**: for backtesting (in `data/` directory)
- **Arctic/MongoDB**: for production (time series database)
- **Parquet**: newer option for file-based storage

The production system fetches daily prices from IB, stores in MongoDB, and runs the full pipeline.

---

## F. Parameter Estimation Methodology

### F.1 What Gets Fitted vs. What's Fixed

| Parameter | Fitted or Fixed? | Method |
|-----------|-----------------|--------|
| Forecast scalar | Fitted (expanding window) | Pooled across instruments, abs median |
| Forecast weights | Either | Handcrafting or fixed equal weights |
| FDM | Fitted | From forecast correlations |
| Instrument weights | Either | Handcrafting or fixed |
| IDM | Fitted | From instrument return correlations |
| Vol lookback (35 days) | Fixed | Not optimised |
| EWMAC spans | Fixed | Chosen from standard set, not optimised |
| Carry smoothing (90 days) | Fixed | Not optimised |
| Risk target (25%) | Fixed | Chosen, not fitted |
| Buffer size (10%) | Fixed | Not optimised |

### F.2 Forecast Scalar Estimation

**Method**: Expanding window with cross-sectional pooling.

```
1. For each rule variation:
   a. Collect raw forecasts across ALL instruments (cross-sectional)
   b. At each point in time, take median of absolute values across instruments
   c. Apply expanding window mean to get avg_abs_value(t)
   d. scalar(t) = 10.0 / avg_abs_value(t)
   e. Backfill the first valid estimate

2. Parameters:
   window = 250000 (effectively expanding, not rolling)
   min_periods = 500 (need 2 years before first estimate)
   backfill = True (use first estimate for earlier period)
```

**Why pool across instruments?**:
- More data points = more robust estimate
- Scalar should be similar across instruments (since vol is already normalized)
- Avoids jumps when new instruments appear

### F.3 Correlation Estimation

```python
# From sysquant/estimators/
# Uses expanding or rolling window
# Correlations estimated at lower frequency (monthly or annual)
# Then smoothed with EWM

# Key parameters:
correlation_estimate:
    func: sysquant.estimators.correlation_estimator.correlationEstimator
    frequency: "M"  # monthly
    date_method: "expanding"  # or "rolling"
    min_periods: 20
    floor_at_zero: True  # correlations floored at 0 for diversification
    cleaning: True  # handle missing data
    using_exponent: True
    ew_lookback: 500  # for exponential weighting
```

### F.4 Avoiding Overfitting -- Carver's Principles

1. **Pool across instruments**: Forecast scalars, correlations are estimated across instruments, not per instrument.

2. **Use very long lookbacks**: min_periods=500 for scalar, expanding windows preferred.

3. **Handcrafting**: Massively inflated uncertainty (fudge factor 4.0) pushes toward equal weights.

4. **Fixed rule specifications**: EWMAC spans are from a standard set (2,4,8,16,32,64). They are NOT optimised.

5. **No parameter scanning**: Carver explicitly warns against testing many parameter combinations. Pick a standard set.

6. **In-sample/out-of-sample split**: Use the first half for estimation, second half for validation. But even better: use expanding window so parameters are always estimated only on past data.

7. **Minimum weight of 5%**: Instruments or rules with < 5% weight should be dropped entirely.

8. **Conservative FDM/IDM caps**: Both capped at 2.5.

9. **Robust vol calculation**: The vol floor prevents unrealistic vol estimates from creating extreme positions.

10. **Cost penalties**: When estimating weights, subtract expected trading costs from returns. This naturally penalises high-turnover rules.

### F.5 What Carver Says About the "Fit Period"

From his blog:
- Correlations and scalars are estimated on an **expanding window** basis
- The expanding window means at time t, you use all data from the start up to t
- There is a minimum data requirement (min_periods) before any estimate is produced
- Before that minimum, a backfilled estimate is used

---

## G. Risk Management Layers

Carver's system has **four layers** of risk management:

### G.1 Layer 1: Endogenous (Position Sizing)

The volatility targeting in position sizing IS risk management:
```
position = vol_scalar * forecast / 10
```

This ensures:
- Each instrument contributes roughly equally to portfolio vol
- Total portfolio vol targets a specific level (e.g., 25%)
- Position sizes automatically decrease when vol increases

### G.2 Layer 2: Forecast Capping

```
capped_forecast = forecast.clip(-20, +20)
```

This limits maximum position to 2x the "average" position (since avg_abs_forecast=10, max forecast=20).

### G.3 Layer 3: Position Inertia (Buffering)

From `systems/buffering.py`:

**Forecast method buffer**:
```python
def _calculate_forecast_buffer_method(position, buffer_size, vol_scalar,
                                       idm=1.0, instr_weight=1.0):
    average_position = abs(vol_scalar * instr_weight * idm)
    buffer = average_position * buffer_size  # default buffer_size = 0.10
    return buffer
```

**How buffering works**:
```
buffer_zone = average_position * 0.10  (10% of average position)
top_pos = optimal_position + buffer_zone
bot_pos = optimal_position - buffer_zone

Rule: Only trade if current_position is outside [bot_pos, top_pos]
      When trading, trade to the nearest edge of the buffer
```

**Purpose**: Reduces unnecessary trading (and costs) from small forecast changes.

**Position method buffer** (alternative):
```python
buffer = abs(position) * buffer_size  # 10% of current position
```

### G.4 Layer 4: Risk Overlay (Exogenous)

From `systems/risk_overlay.py`. This is a portfolio-level multiplier between 0 and 1.

**Four components**, taking the minimum (most conservative):

```python
risk_multiplier = min(
    risk_multiplier_for_normal_risk,      # expected portfolio risk vs limit
    risk_multiplier_for_shocked_stdev,    # vol-shocked risk vs limit
    risk_multiplier_for_sum_abs_risk,     # correlation shock (sum of abs positions)
    risk_multiplier_for_leverage          # total leverage vs limit
)
```

**Component 1: Normal Risk**
```
normal_risk = sqrt(W * Sigma * W^T)  # portfolio vol using recent correlations
risk_limit = max_risk_fraction_normal_risk * vol_target
multiplier = risk_limit / max(risk_limit, normal_risk)
```

**Component 2: Shocked Vol Risk**
```
# Use max(recent_vol, 2*recent_vol) for each instrument
# Recalculate portfolio risk
# This catches "vol is about to double" scenarios
```

**Component 3: Sum of Absolute Positions Risk**
```
sum_abs_risk = sum(abs(position_i * instrument_vol_i))
# This is the portfolio risk assuming all correlations go to 1
risk_limit = max_risk_limit_sum_abs_risk * vol_target
```

**Component 4: Leverage**
```
leverage = sum(abs(notional_exposure_i)) / capital
risk_limit = max_risk_leverage  # e.g., 10x
```

**Default config**:
```yaml
risk_overlay:
    max_risk_fraction_normal_risk: 2.0    # max 2x target vol
    max_risk_fraction_stdev_risk: 4.0     # max 4x target vol (shocked)
    max_risk_limit_sum_abs_risk: 6.0      # max 6x target vol (all corr=1)
    max_risk_leverage: 10.0               # max 10x leverage
```

**How it works in practice**:
- Under normal conditions, the multiplier is 1.0 (no reduction)
- During stress: gradually reduces all positions proportionally
- This is the "circuit breaker" for when the system might blow up

---

## H. Recommended Project Structure

### H.1 pysystemtrade's Actual Structure

```
pysystemtrade/
    systems/                    # The backtesting engine
        basesystem.py           # System class - assembles stages
        stage.py                # SystemStage base class
        system_cache.py         # Caching decorators (@input, @output, @diagnostic)
        rawdata.py              # Stage 1: prices, vol, carry preprocessing
        trading_rules.py        # TradingRule class
        forecasting.py          # Stage 2: Rules stage (applies TradingRule objects)
        forecast_scale_cap.py   # Stage 3: scale and cap forecasts
        forecast_combine.py     # Stage 4: combine forecasts with weights + FDM
        positionsizing.py       # Stage 5: vol targeting, subsystem positions
        portfolio.py            # Stage 6: instrument weights, IDM, risk overlay
        buffering.py            # Position buffer calculations
        risk_overlay.py         # Risk multiplier calculations
        forecast_mapping.py     # Non-linear forecast mapping (optional)
        provided/               # Pre-built systems and rules
            rules/              # Trading rule implementations
                ewmac.py
                breakout.py
                carry.py
                accel.py
                rel_mom.py
                cs_mr.py
                mr_wings.py
            example/            # Example system configs
            futures_chapter15/  # Book chapter 15 system
            rob_system/         # Rob's actual system
        accounts/               # P&L and account curve calculations
        tools/                  # Utilities (autogroup, etc.)
        tests/                  # Unit tests

    sysquant/                   # Quantitative calculations
        estimators/             # Statistical estimators
            vol.py              # robust_vol_calc, mixed_vol_calc
            forecast_scalar.py  # Forecast scalar estimation
            correlations.py     # Correlation estimation
            diversification_multipliers.py  # FDM/IDM calculation
            stdev_estimator.py  # Standard deviation estimation
            turnover.py         # Turnover estimation
        optimisation/           # Weight optimisation
            full_handcrafting.py    # Handcrafting algorithm
            generic_optimiser.py   # Optimiser framework
            weights.py             # portfolioWeights class
            pre_processing.py      # Returns pre-processing
        returns.py              # Return series objects
        portfolio_risk.py       # Portfolio risk calculations

    sysdata/                    # Data handling
        sim/                    # Simulation data sources
            csv_futures_sim_data.py   # CSV-based data
            futures_sim_data.py       # Base class
        config/                 # Configuration
            configdata.py       # Config class
        parquet/                # Parquet storage
        arctic/                 # MongoDB/Arctic storage

    sysobjects/                 # Domain objects
        carry_data.py           # Carry data object
        contracts.py            # Contract specifications

    sysbrokers/                 # Broker integrations
        IB/                     # Interactive Brokers

    sysproduction/              # Live trading system
        run_systems.py          # Run backtest in production
        run_stack_handler.py    # Order execution
        run_daily_price_updates.py  # Daily data updates
        update_fx_prices.py
        update_historical_prices.py
        strategy_code/          # Strategy-specific code

    sysexecution/               # Order management
    syscontrol/                 # Process management
    syslogdiag/                 # Logging and diagnostics
    data/                       # CSV data files
        futures/
            adjusted_prices_csv/
            multiple_prices_csv/
            fx_prices_csv/
```

### H.2 Recommended Structure for a New CTA System

Based on the patterns in pysystemtrade, adapted for production use:

```
my_cta_system/
    config/
        default.yaml            # Default system parameters
        instruments.yaml        # Instrument specifications
        trading_rules.yaml      # Rule definitions

    data/
        providers/              # Data source adapters
            ib_data.py          # Interactive Brokers
            csv_data.py         # CSV files
            parquet_data.py     # Parquet storage
        pipeline/
            contract_prices.py  # Individual contract prices
            multiple_prices.py  # Multi-price data (price, carry, forward)
            adjusted_prices.py  # Back-adjusted continuous prices
            roll_calendar.py    # Roll date management
            fx_rates.py         # FX rate handling

    indicators/                 # Trading rules / signals
        ewmac.py                # EWMAC trend following
        breakout.py             # Breakout / channel
        carry.py                # Carry signal
        momentum.py             # Relative momentum
        mean_reversion.py       # Mean reversion signals
        acceleration.py         # Acceleration signals

    pipeline/                   # The processing stages
        rawdata.py              # Stage 1: preprocessing
        forecast.py             # Stage 2: raw forecasts
        scaling.py              # Stage 3: forecast scalar + cap
        combining.py            # Stage 4: forecast combination + FDM
        position_sizing.py      # Stage 5: vol targeting
        portfolio.py            # Stage 6: instrument weights + IDM

    risk/
        volatility.py           # Robust vol estimation
        risk_overlay.py         # Portfolio risk overlay
        buffering.py            # Position buffer logic
        drawdown.py             # Drawdown monitoring

    optimizer/
        handcrafting.py         # Handcrafting weight estimation
        correlation.py          # Correlation estimation
        diversification.py      # FDM/IDM calculation
        forecast_scalar.py      # Forecast scalar estimation

    portfolio/
        construction.py         # Portfolio construction
        rebalancing.py          # Rebalancing logic

    execution/                  # Live trading
        order_manager.py        # Order generation
        broker_adapter.py       # Broker interface
        position_reconciliation.py  # Position checks

    monitoring/
        performance.py          # P&L tracking
        risk_report.py          # Risk reports
        alerting.py             # Alerts

    research/                   # Research notebooks
        backtest.py             # Backtesting scripts

    tests/
        test_indicators/
        test_pipeline/
        test_risk/
        test_optimizer/
```

### H.3 Key Design Principles

1. **Stages are independent and composable**: Each stage depends only on its inputs, not on the implementation of other stages.

2. **Caching is critical**: Every intermediate calculation should be cached. pysystemtrade uses decorators (`@output()`, `@diagnostic()`, `@input`). In a new system, use `functools.lru_cache` or a custom cache.

3. **Config-driven**: Rules, parameters, instruments are all defined in YAML config, not hardcoded.

4. **Separation of backtest and production**: The same pipeline should run in both modes, with only the data source changing.

5. **Immutable data flow**: Each stage produces new data, doesn't modify inputs.

6. **Per-instrument parallelism**: Most calculations are independent per instrument and can be parallelised.

### H.4 Live Trading Integration

pysystemtrade's production system (`sysproduction/`) runs daily:

```
1. update_fx_prices.py          -- fetch FX rates from IB
2. update_historical_prices.py  -- fetch contract prices from IB
3. update_multiple_adjusted_prices.py  -- build/update continuous series
4. run_systems.py               -- run the full backtest pipeline
5. run_strategy_order_generator.py  -- compare desired vs actual positions
6. run_stack_handler.py         -- execute orders via IB
7. run_capital_update.py        -- update capital based on P&L
8. run_reports.py               -- generate daily reports
```

**Key difference from backtest**: In production, positions are not just numbers -- they need to:
- Be compared against current broker positions
- Generate actual orders (market or limit)
- Handle partial fills
- Deal with roll transitions in real contracts
- Account for margin requirements

### H.5 The System Cache Architecture

The caching system uses three decorator levels:

```python
@input      # Data from external sources or other stages. Cached.
@diagnostic # Intermediate calculations. Cached. Can be deleted to save memory.
@output     # Final stage outputs. Cached. Protected from deletion.
@dont_cache # Never cached. Recomputed every time (e.g., switching logic).
```

Cache key structure: `(stage_name, method_name, instrument_code)`

This means calling `system.rawdata.daily_returns_volatility("CRUDE_W")` twice returns the cached result the second time.

---

## Appendix: Carver's Complete System Configuration

From his blog "My trading system" (2021), Carver's actual system uses:

**Rules**:
```
breakout10, breakout20, breakout40, breakout80, breakout160, breakout320
ewmac2, ewmac4, ewmac8, ewmac16, ewmac32, ewmac64
carry10, carry30, carry60, carry125
relmomentum10, relmomentum20, relmomentum40, relmomentum80
mrinasset160
accel2, accel4, accel8, accel16, accel32, accel64
```

**Forecast Sharpe ratios** (approximate, from his blog):
```
breakout: 0.06 to 0.79 (worse at short lookbacks)
ewmac: 0.20 to 0.55 (best at medium lookbacks 16-32)
carry: 0.90 to 0.95 (consistently the best)
relmomentum: -1.86 to 0.13 (poor at short lookbacks)
mrinasset: -0.63 (negative!)
accel: variable
```

**Key configuration values**:
```yaml
percentage_vol_target: 25
notional_trading_capital: 500000
base_currency: USD
average_absolute_forecast: 10.0
forecast_cap: 20.0
buffer_method: forecast
buffer_size: 0.10

volatility_calculation:
  func: sysquant.estimators.vol.robust_vol_calc
  days: 35
  min_periods: 10
  vol_floor: True

risk_overlay:
  max_risk_fraction_normal_risk: 2.0
  max_risk_fraction_stdev_risk: 4.0
  max_risk_limit_sum_abs_risk: 6.0
  max_risk_leverage: 10.0
```

---

## Appendix: Other Open-Source CTA Systems

| System | Language | Notes |
|--------|----------|-------|
| **pysystemtrade** | Python | Gold standard. Full backtest + production. |
| **Zipline** (Quantopian, archived) | Python | Event-driven. Better for equities. No futures roll handling. |
| **backtrader** | Python | Event-driven. Good for prototyping, weak on portfolio-level risk. |
| **vectorbt** | Python | Vectorized backtest. Fast but no production system. |
| **QuantConnect/Lean** | C#/Python | Cloud-based. Has futures support. Complex. |
| **systematictradingexamples** | Python | Carver's book code examples. Good for learning formulas. |

**Recommendation**: Use pysystemtrade as the reference architecture, but build your own system for production. pysystemtrade's codebase is large and has accumulated complexity from years of organic growth. The core concepts (7 stages, caching, config-driven rules) are the valuable parts.

---

## Quick Reference: The Complete Position Formula

```
position(instrument_i, time_t) =

    sum_over_rules(
        forecast_scalar(rule_r)
        * raw_forecast(rule_r, instrument_i, t)
    ).clip(-20, +20)
    * forecast_weight(rule_r)
    (renormalized where forecasts exist)
    * FDM(t)
    .clip(-20, +20)

    * (Capital * VolTarget%) / (sqrt(256) * InstrValueVol(i,t))
    / 10

    * instrument_weight(i)
    * IDM(t)
    * risk_scalar(t)

    then buffered: only trade if outside [position - buffer, position + buffer]

where:
    InstrValueVol = DenomPrice * PointValue * 0.01 * DailyPctVol * FXrate
    FDM = min(1/sqrt(W_f * H_f * W_f^T), 2.5)
    IDM = min(1/sqrt(W_i * H_i * W_i^T), 2.5)
    risk_scalar = min(normal_risk_mult, shocked_vol_mult, sum_abs_mult, leverage_mult)
    buffer = abs(vol_scalar * instr_weight * IDM) * buffer_size
```
