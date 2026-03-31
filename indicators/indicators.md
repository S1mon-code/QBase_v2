# QBase_v2 指标库（324 个）

10 大类 324 个指标，纯 numpy 函数（numpy in → numpy out）。从 QBase v1 迁移并重新分类。

## 分类总览

| 分类 | 数量 | 说明 | 依赖 |
|------|:----:|------|------|
| **momentum** | 35 | 动量/振荡类（RSI, MACD, TSMOM, CCI 等） | numpy |
| **trend** | 35 | 趋势/均线类（EMA, ADX, SuperTrend, KAMA 等） | numpy |
| **volatility** | 41 | 波动率/统计类（ATR, Bollinger, Hurst, Yang-Zhang, Z-Score 等） | numpy |
| **volume** | 38 | 量价/资金流（OBV, CMF, VWAP, Volume Momentum, OI 等） | numpy |
| **ml** | 61 | 机器学习/统计学习（HMM, KMeans, PCA, Kalman, GARCH 等） | numpy + sklearn |
| **regime** | 35 | 行情状态识别（Hurst Regime, Entropy, Jump, Choppiness 等） | numpy |
| **spread** | 25 | 价差/跨品种（Basis, Carry, Beta, Ratio 等） | numpy |
| **structure** | 24 | 市场结构/OI（OI Flow, Smart Money, Commitment 等） | numpy |
| **microstructure** | 15 | 微观结构（Amihud, Kyle Lambda, Spread Proxy 等） | numpy |
| **seasonality** | 15 | 季节性（月效应, 周效应, 到期日, 季节分解等） | numpy |

## 共享工具

`_utils.py` 提供共享 helper 函数，避免跨文件重复：
- `_ema(arr, period)` — 指数移动平均
- `_ema_skip_nan(arr, period)` — 跳过 NaN 的 EMA
- `_sma(arr, period)` — 简单移动平均
- `_rsi(closes, period)` — RSI（Wilder's smoothing）

---

## Momentum（35 个）

| 指标 | 文件 | 函数 |
|------|------|------|
| Awesome Oscillator | `awesome_oscillator.py` | `ao(highs, lows, fast, slow)` |
| CCI | `cci.py` | `cci(highs, lows, closes, period)` |
| Center of Gravity | `center_of_gravity.py` | `cog(closes, period)` |
| Close Location | `close_location.py` | `close_location(highs, lows, closes)` |
| CMO | `cmo.py` | `cmo(closes, period)` |
| Connors RSI | `connors_rsi.py` | `connors_rsi(closes, rsi_period, streak_period, pct_rank_period)` |
| Coppock | `coppock.py` | `coppock(closes, wma_period, roc_long, roc_short)` |
| Cyber Cycle | `cyber_cycle.py` | `cyber_cycle(closes, alpha)` |
| DPO | `dpo.py` | `dpo(closes, period)` |
| Elder Force Index | `elder_force.py` | `elder_force_index(closes, volumes, period)` |
| Ergodic | `ergodic.py` | `ergodic(closes, short_period, long_period, signal_period)` |
| Fisher Transform | `fisher_transform.py` | `fisher_transform(highs, lows, period)` |
| KST | `kst.py` | `kst(closes, roc_periods, sma_periods, signal_period)` |
| MACD | `macd.py` | `macd(closes, fast, slow, signal)` |
| Momentum Acceleration | `momentum_accel.py` | `momentum_acceleration(closes, fast_period, slow_period)` |
| PPO | `ppo.py` | `ppo(closes, fast, slow, signal)` |
| Pretty Good Oscillator | `pretty_good_oscillator.py` | `pgo(closes, highs, lows, period)` |
| Price Acceleration | `price_acceleration.py` | `price_acceleration(closes, period)` |
| Price Momentum Quality | `price_momentum_quality.py` | `pmq(closes, period)` |
| Reflex | `reflex.py` | `reflex(closes, period)` |
| Relative Momentum Index | `relative_momentum_index.py` | `rmi(closes, period, lookback)` |
| Relative Vigor Index | `relative_vigor.py` | `relative_vigor_index(opens, highs, lows, closes, period)` |
| ROC | `roc.py` | `rate_of_change(closes, period)` |
| Rocket RSI | `rocket_rsi.py` | `rocket_rsi(closes, rsi_period, rocket_period)` |
| RSI | `rsi.py` | `rsi(closes, period)` |
| Schaff Trend Cycle | `schaff_trend.py` | `schaff_trend_cycle(closes, period, fast, slow)` |
| SMA Ratio Z-Score | `sma_ratio_zscore.py` | `sma_ratio_zscore(closes, lookback)` |
| Stochastic RSI | `stoch_rsi.py` | `stoch_rsi(closes, rsi_period, stoch_period, k_period, d_period)` |
| Stochastic / KDJ | `stochastic.py` | `stochastic(highs, lows, closes, k_period, d_period)` / `kdj(...)` |
| TrendFlex | `trend_flex.py` | `trendflex(closes, period)` |
| TRIX | `trix.py` | `trix(closes, period)` |
| TSI | `tsi.py` | `tsi(closes, long_period, short_period, signal_period)` |
| TSMOM | `tsmom.py` | `tsmom(closes, lookback, vol_lookback)` |
| Ultimate Oscillator | `ultimate_oscillator.py` | `ultimate_oscillator(highs, lows, closes, p1, p2, p3)` |
| Williams %R | `williams_r.py` | `williams_r(highs, lows, closes, period)` |

## Trend（35 个）

| 指标 | 文件 | 函数 |
|------|------|------|
| ADX | `adx.py` | `adx(highs, lows, closes, period)` / `adx_with_di(...)` |
| ALMA | `alma.py` | `alma(data, period, offset, sigma)` |
| Aroon | `aroon.py` | `aroon(highs, lows, period)` |
| Decycler | `decycler.py` | `decycler(closes, period)` |
| DEMA | `dema.py` | `dema(data, period)` |
| Donchian | `donchian.py` | `donchian(highs, lows, period)` |
| Ehlers Instantaneous | `ehlers_instantaneous.py` | `instantaneous_trendline(closes, alpha)` |
| EMA | `ema.py` | `ema(data, period)` / `ema_cross(data, fast, slow)` |
| EMA Ribbon | `ema_ribbon.py` | `ema_ribbon(data, periods)` / `ema_ribbon_signal(...)` |
| Fractal | `fractal.py` | `fractal_high(highs, period)` / `fractal_low(...)` / `fractal_levels(...)` |
| FRAMA | `frama.py` | `frama(closes, period)` |
| Higher Low / Lower High | `higher_low.py` | `higher_lows(lows, lookback)` / `lower_highs(highs, lookback)` |
| HMA | `hma.py` | `hma(data, period)` |
| Ichimoku | `ichimoku.py` | `ichimoku(highs, lows, closes, tenkan, kijun, senkou_b, displacement)` |
| Jurik MA | `jurik_ma.py` | `jma(closes, period, phase, power)` |
| KAMA | `kama.py` | `kama(closes, period, fast_sc, slow_sc)` |
| Kaufman ER Bands | `kaufman_er_bands.py` | `er_bands(closes, period, mult)` |
| Keltner | `keltner.py` | `keltner(highs, lows, closes, ema_period, atr_period, multiplier)` |
| Laguerre Filter | `laguerre_filter.py` | `laguerre(closes, gamma)` |
| Linear Regression | `linear_regression.py` | `linear_regression(data, period)` / `linear_regression_slope(...)` / `r_squared(...)` |
| Mass Index | `mass_index.py` | `mass_index(highs, lows, ema_period, sum_period)` |
| McGinley Dynamic | `mcginley.py` | `mcginley_dynamic(data, period)` |
| MESA Adaptive MA | `mesa_adaptive_ma.py` | `mama(closes, fast_limit, slow_limit)` |
| Parabolic SAR | `psar.py` | `psar(highs, lows, af_start, af_step, af_max)` |
| Random Walk Index | `rwi.py` | `rwi(highs, lows, closes, period)` |
| Ehlers Sine Wave | `sine_wave.py` | `ehlers_sine_wave(closes, alpha)` |
| SMA | `sma.py` | `sma(data, period)` |
| SuperTrend | `supertrend.py` | `supertrend(highs, lows, closes, period, multiplier)` |
| T3 | `t3.py` | `t3(data, period, volume_factor)` |
| TEMA | `tema.py` | `tema(data, period)` |
| Trend Intensity | `trend_intensity.py` | `trend_intensity(closes, period)` |
| VIDYA | `vidya.py` | `vidya(closes, period, cmo_period)` |
| Vortex | `vortex.py` | `vortex(highs, lows, closes, period)` |
| VWMA | `vwma.py` | `vwma(closes, volumes, period)` |
| ZLEMA | `zlema.py` | `zlema(data, period)` |

## Volatility（41 个）

| 指标 | 文件 | 函数 |
|------|------|------|
| Acceleration Bands | `acceleration_bands.py` | `acceleration_bands(highs, lows, closes, period, width)` |
| ADR | `adr.py` | `average_day_range(highs, lows, period)` / `adr_percent(...)` |
| ATR | `atr.py` | `atr(highs, lows, closes, period)` |
| ATR Ratio | `atr_ratio.py` | `atr_ratio(highs, lows, closes, short_period, long_period)` |
| Bollinger Bands | `bollinger.py` | `bollinger_bands(closes, period, num_std)` / `bollinger_width(...)` |
| Chaikin Volatility | `chaikin_vol.py` | `chaikin_volatility(highs, lows, ema_period, roc_period)` |
| Chandelier Exit | `chandelier_exit.py` | `chandelier_exit(highs, lows, closes, period, multiplier)` |
| Close-to-Close Vol | `close_to_close_vol.py` | `close_to_close_vol(closes, period, annualize)` |
| Conditional Volatility | `conditional_vol.py` | `conditional_volatility(closes, period, threshold)` |
| Entropy | `entropy.py` | `entropy(closes, period, bins)` |
| Garman-Klass | `garman_klass.py` | `garman_klass(opens, highs, lows, closes, period)` |
| GK Volatility Ratio | `gk_volatility_ratio.py` | `gk_vol_ratio(opens, highs, lows, closes, fast, slow)` |
| Historical Kurtosis | `historical_kurtosis.py` | `rolling_kurtosis(returns_or_closes, period)` |
| Historical Skewness | `historical_skew.py` | `rolling_skewness(returns_or_closes, period)` |
| Historical Volatility | `historical_vol.py` | `historical_volatility(closes, period, annualize)` |
| Hurst Exponent | `hurst.py` | `hurst_exponent(data, max_lag)` |
| Information Ratio | `information_ratio.py` | `rolling_information_ratio(returns, benchmark_returns, period)` |
| Keltner Width | `keltner_width.py` | `keltner_width(highs, lows, closes, ema_period, atr_period, multiplier)` |
| NATR | `natr.py` | `natr(highs, lows, closes, period)` |
| Normalized Range | `normalized_range.py` | `normalized_range(highs, lows, closes, period)` |
| NR7 / NR4 | `nr7.py` | `nr7(highs, lows)` / `nr4(highs, lows)` |
| Parkinson | `parkinson.py` | `parkinson(highs, lows, period)` |
| Price Density | `price_density.py` | `price_density(highs, lows, closes, period)` |
| Quantile Bands | `quantile_bands.py` | `quantile_regression_bands(closes, period, quantiles)` |
| Range Expansion | `range_expansion.py` | `range_expansion(highs, lows, closes, period)` |
| Range Expansion Signal | `range_expansion_signal.py` | `range_expansion_signal(opens, highs, lows, closes, ...)` |
| Realized Skewness | `realized_skew.py` | `realized_skewness(highs, lows, closes, period)` |
| Realized Variance | `realized_variance.py` | `realized_variance(closes, period)` / `realized_volatility(...)` |
| Relative Volatility | `relative_vol.py` | `relative_volatility(closes, fast, slow)` |
| Robust Z-Score | `robust_zscore.py` | `robust_zscore(data, period)` |
| Rogers-Satchell | `rogers_satchell.py` | `rogers_satchell(opens, highs, lows, closes, period)` |
| Rolling Percentile Rank | `rolling_percentile_rank.py` | `percentile_rank_features(features_matrix, period)` |
| Rolling Std Dev / Z-Score | `std_dev.py` | `rolling_std(data, period)` / `z_score(data, period)` |
| Successive Differences | `successive_differences.py` | `von_neumann_ratio(data, period)` |
| True Range | `true_range.py` | `true_range(highs, lows, closes)` |
| TTM Squeeze | `ttm_squeeze.py` | `ttm_squeeze(highs, lows, closes, bb_period, bb_mult, kc_period, kc_mult)` |
| Ulcer Index | `ulcer_index.py` | `ulcer_index(closes, period)` |
| Vol of Vol Regime | `vol_of_vol_regime.py` | `vol_of_vol_regime(closes, vol_period, vov_period)` |
| Volatility Ratio | `vol_ratio.py` | `volatility_ratio(highs, lows, closes, period)` |
| Vol of Vol | `vov.py` | `vov(closes, vol_period, vov_period)` |
| Yang-Zhang | `yang_zhang.py` | `yang_zhang(opens, highs, lows, closes, period)` |

## Volume（38 个）

| 指标 | 文件 | 函数 |
|------|------|------|
| A/D Line | `ad_line.py` | `ad_line(highs, lows, closes, volumes)` |
| Buying/Selling Pressure | `buying_selling_pressure.py` | `buying_selling_pressure(highs, lows, closes, volumes, period)` |
| Chaikin Oscillator | `chaikin_oscillator.py` | `chaikin_oscillator(highs, lows, closes, volumes, fast, slow)` |
| CMF | `cmf.py` | `cmf(highs, lows, closes, volumes, period)` |
| Demand Index | `demand_index.py` | `demand_index(highs, lows, closes, volumes, period)` |
| EMV Signal | `ease_of_movement_signal.py` | `emv_signal(highs, lows, volumes, period, signal_period)` |
| EMV | `emv.py` | `emv(highs, lows, volumes, period)` |
| Force Index | `force_index.py` | `force_index(closes, volumes, period)` |
| Intraday Intensity | `intraday_intensity.py` | `intraday_intensity(highs, lows, closes, volumes, period)` |
| Klinger | `klinger.py` | `klinger(highs, lows, closes, volumes, fast, slow, signal)` |
| MFI | `mfi.py` | `mfi(highs, lows, closes, volumes, period)` |
| Money Flow | `money_flow.py` | `money_flow(highs, lows, closes, volumes)` / `money_flow_ratio(...)` |
| Normalized Volume | `normalized_volume.py` | `normalized_volume(volumes, period)` |
| NVI / PVI | `nvi.py` | `nvi(closes, volumes)` / `pvi(closes, volumes)` |
| OBV | `obv.py` | `obv(closes, volumes)` |
| OI Accumulation | `oi_accumulation.py` | `oi_accumulation(closes, oi, period)` |
| OI Adjusted Volume | `oi_adjusted_volume.py` | `oi_adjusted_volume(volumes, oi, period)` |
| OI Climax | `oi_climax.py` | `oi_climax(oi, volumes, period, threshold)` |
| OI Divergence | `oi_divergence.py` | `oi_divergence(closes, oi, period)` |
| OI Flow | `oi_flow.py` | `oi_flow(closes, oi, volumes, period)` |
| OI Momentum | `oi_momentum.py` | `oi_momentum(oi, period)` / `oi_sentiment(...)` |
| OI Rate of Change | `oi_rate_of_change.py` | `oi_roc(oi, period)` |
| PVI Signal | `positive_volume_index_signal.py` | `pvi_signal(closes, volumes, period)` |
| Trade Volume Index | `trade_volume_index.py` | `tvi(closes, volumes, min_tick)` |
| Twiggs Money Flow | `twiggs.py` | `twiggs_money_flow(highs, lows, closes, volumes, period)` |
| Volume Efficiency | `volume_efficiency.py` | `volume_efficiency(closes, volumes, period)` |
| Volume Force | `volume_force.py` | `volume_force(closes, volumes, period)` |
| Volume Momentum | `volume_momentum.py` | `volume_momentum(volumes, period)` / `relative_volume(...)` |
| Volume Oscillator | `volume_oscillator.py` | `volume_oscillator(volumes, fast, slow)` |
| VPT | `volume_price_trend.py` | `vpt(closes, volumes)` |
| Volume Profile | `volume_profile.py` | `volume_profile(closes, volumes, bins)` / `poc(...)` |
| Volume Spike | `volume_spike.py` | `volume_spike(volumes, period, threshold)` / `volume_climax(...)` / `volume_dry_up(...)` |
| VW-MACD | `volume_weighted_macd.py` | `vwmacd(closes, volumes, fast, slow, signal)` |
| VW-RSI | `volume_weighted_rsi.py` | `volume_rsi(closes, volumes, period)` |
| VROC | `vroc.py` | `vroc(volumes, period)` |
| VWAP | `vwap.py` | `vwap(highs, lows, closes, volumes)` / `vwap_session(...)` |
| WAD | `wad.py` | `wad(highs, lows, closes)` |
| Wyckoff Divergence | `wyckoff_divergence.py` | `wyckoff_divergence(highs, lows, closes, volumes, lookback)` |

## ML（61 个）

| 指标 | 文件 | 函数 | 依赖 |
|------|------|------|------|
| Adaptive LMS | `adaptive_lms.py` | `lms_filter(closes, reference, period, mu)` | numpy |
| Attention Score | `attention_score.py` | `attention_weights(features_matrix, target, period)` | numpy |
| Autoencoder Error | `autoencoder_error.py` | `reconstruction_error(features_matrix, period, encoding_dim)` | sklearn |
| Bayesian Trend | `bayesian_trend.py` | `bayesian_online_trend(closes, hazard_rate)` | numpy |
| Boosting Signal | `boosting_signal.py` | `gradient_boost_signal(closes, features_matrix, period, n_estimators)` | sklearn |
| Copula Tail | `copula_tail.py` | `tail_dependence(returns_a, returns_b, period, quantile)` | numpy |
| CV Signal | `cross_validation_signal.py` | `cv_signal_strength(closes, features_matrix, period, n_folds)` | sklearn |
| CUSUM Filter | `cusum_filter.py` | `cusum_event_filter(closes, threshold)` | numpy |
| Decision Boundary | `decision_boundary.py` | `decision_boundary_distance(features_matrix, labels, period)` | sklearn |
| Disagreement | `disagreement_index.py` | `model_disagreement(closes, features_matrix, period)` | sklearn |
| DTW | `dynamic_time_warping.py` | `dtw_distance(series_a, series_b, period)` | numpy |
| Elastic Net | `elastic_net_forecast.py` | `elastic_net_signal(closes, features_matrix, period, alpha, l1_ratio)` | sklearn |
| Ensemble Vote | `ensemble_signal.py` | `ensemble_vote(closes, features_matrix, period)` | sklearn |
| Multi Entropy | `entropy_features.py` | `multi_entropy(closes, period)` | scipy |
| Feature Importance | `feature_importance.py` | `rolling_tree_importance(closes, features_matrix, period, n_estimators)` | sklearn |
| Fractal Market | `fractal_market.py` | `fractal_market_hypothesis(closes, period)` | numpy |
| GMM Regime | `gaussian_mixture_regime.py` | `gmm_regime(features_matrix, period, n_components)` | sklearn |
| Gradient Signal | `gradient_trend.py` | `gradient_signal(closes, period, smoothing)` | numpy |
| Granger Proxy | `granger_proxy.py` | `granger_causality_score(series_a, series_b, period, max_lag)` | numpy |
| Hilbert Features | `hilbert_features.py` | `hilbert_transform_features(closes, period)` | scipy |
| HMM Regime | `hmm_regime.py` | `hmm_regime(closes, n_states, period)` | numpy |
| Incremental PCA | `incremental_pca.py` | `incremental_pca_signal(features_matrix, n_components)` | sklearn |
| Isolation Anomaly | `isolation_anomaly.py` | `isolation_anomaly(features_matrix, period, contamination)` | sklearn |
| Adaptive Kalman | `kalman_adaptive.py` | `adaptive_kalman(closes, period)` | numpy |
| Kalman Filter | `kalman_trend.py` | `kalman_filter(closes, process_noise, measurement_noise)` | numpy |
| KDE Levels | `kernel_density_levels.py` | `kde_support_resistance(closes, period, n_levels)` | numpy |
| K-Means Regime | `kmeans_regime.py` | `kmeans_regime(features_matrix, period, n_clusters)` | sklearn |
| Lasso Importance | `lasso_importance.py` | `rolling_lasso_importance(closes, features_matrix, period)` | sklearn |
| Manifold Embedding | `manifold_features.py` | `manifold_embedding(features_matrix, period, n_components)` | sklearn |
| Momentum Decompose | `momentum_decompose.py` | `momentum_components(closes, short, medium, long)` | numpy |
| Mutual Information | `mutual_information.py` | `rolling_mutual_info(series_a, series_b, period, n_bins)` | numpy |
| KNN Signal | `nearest_neighbor_signal.py` | `knn_signal(closes, features_matrix, period, k)` | sklearn |
| Distance Correlation | `nonlinear_correlation.py` | `distance_correlation(series_a, series_b, period)` | numpy |
| OI Anomaly | `oi_anomaly.py` | `oi_anomaly(closes, oi, volumes, period)` | sklearn |
| OI Cluster | `oi_cluster.py` | `oi_cluster(closes, oi, volumes, period, n_clusters)` | sklearn |
| OI Kalman | `oi_kalman.py` | `oi_kalman_trend(oi, process_noise, measurement_noise)` | numpy |
| OI PCA | `oi_pca.py` | `oi_pca_features(closes, oi, volumes, period)` | sklearn |
| OI Prediction | `oi_prediction.py` | `oi_predicted_return(closes, oi, volumes, period)` | sklearn |
| Online Covariance | `online_covariance.py` | `exponential_covariance(returns_matrix, halflife)` | numpy |
| Welford Stats | `online_mean_variance.py` | `welford_stats(data)` | numpy |
| Online SGD | `online_regression.py` | `online_sgd_signal(closes, features_matrix, learning_rate, period)` | numpy |
| O-U Model | `ou_model.py` | `ou_params(closes, period, dt)` / `ou_deviation(...)` | numpy |
| Rolling PCA | `pca_features.py` | `rolling_pca(features_matrix, period, n_components)` | sklearn |
| Piecewise Trend | `prophet_like_trend.py` | `piecewise_trend(closes, n_changepoints, period)` | numpy |
| Random Projection | `random_projection.py` | `random_projection_features(features_matrix, n_components, period)` | sklearn |
| Recurrence Rate | `recurrence.py` | `recurrence_rate(data, period, threshold_pct)` | numpy |
| Regime Duration | `regime_persistence.py` | `regime_duration(regime_labels, period)` | numpy |
| Transition Matrix | `regime_transition_matrix.py` | `transition_features(regime_labels, n_states)` | numpy |
| Ridge Forecast | `ridge_forecast.py` | `rolling_ridge(closes, features_matrix, period, forecast_horizon)` | sklearn |
| RLS Filter | `rls_filter.py` | `rls_filter(closes, order, forgetting)` | numpy |
| Rolling Eigen | `rolling_correlation_matrix.py` | `rolling_eigen_features(features_matrix, period)` | numpy |
| Correlation Network | `rolling_correlation_network.py` | `correlation_network_score(returns_matrix, period, threshold)` | numpy |
| Ledoit-Wolf | `shrinkage_covariance.py` | `ledoit_wolf_features(returns_matrix, period)` | sklearn |
| Spectral Regime | `spectral_clustering_regime.py` | `spectral_regime(features_matrix, period, n_clusters)` | sklearn |
| Symbolic Features | `symbolic_regression_signal.py` | `symbolic_features(closes, highs, lows, volumes, period)` | numpy |
| Target Encoding | `target_encoding.py` | `target_encoded_regime(closes, period, n_bins)` | numpy |
| Transfer Entropy | `transfer_entropy.py` | `transfer_entropy(source, target, period, n_bins, lag)` | numpy |
| L1 Trend Filter | `trend_filter.py` | `l1_trend_filter(closes, lambda_val)` | numpy |
| Variational Regime | `variational_inference.py` | `variational_regime(closes, period, n_components)` | sklearn |
| GARCH Forecast | `volatility_forecast.py` | `garch_like_forecast(closes, period, alpha, beta)` | numpy |
| Wavelet Decompose | `wavelet_decompose.py` | `wavelet_features(closes, wavelet, level)` | numpy |

## Regime（35 个）

| 指标 | 文件 | 函数 |
|------|------|------|
| Adaptive Lookback | `adaptive_period.py` | `adaptive_lookback(closes, min_period, max_period)` |
| Changepoint | `changepoint.py` | `changepoint_score(data, period)` |
| Choppiness Index | `choppiness_index.py` | `choppiness_index(highs, lows, closes, period)` |
| Complexity Profile | `complexity_profile.py` | `complexity_profile(closes, scales)` |
| Correlation Breakdown | `correlation_breakdown.py` | `correlation_breakdown(closes_a, closes_b, period, stress_threshold)` |
| Distribution Shift | `distribution_shift.py` | `kl_divergence_shift(data, period, reference_period)` |
| Efficiency Ratio | `efficiency_ratio.py` | `efficiency_ratio(closes, period)` |
| Entropy Rate | `entropy_rate.py` | `entropy_rate(closes, period, m)` |
| Fractal Dimension | `fractal_dimension.py` | `fractal_dim(data, period)` |
| Hurst R/S | `hurst_rs.py` | `hurst_rs(data, min_period, max_period)` |
| Jump Detector | `jump_detector.py` | `jump_detection(closes, period, threshold)` |
| Macro Regime Filter | `macro_regime_filter.py` | `macro_regime_filter(closes, volumes, ma_period, ...)` |
| Market State | `market_state.py` | `market_state(closes, volumes, oi, period)` |
| Mean Crossing Rate | `mean_crossing_rate.py` | `mean_crossing(closes, period)` |
| Mean Reversion Speed | `mean_reversion_speed.py` | `ou_speed(data, period)` |
| Mean-Variance Regime | `mean_variance_regime.py` | `mv_regime(data, period, n_regimes)` |
| Momentum Regime | `momentum_regime.py` | `momentum_regime(closes, fast, slow)` |
| OI Cycle | `oi_cycle.py` | `oi_cycle(oi, period)` |
| OI Regime | `oi_regime.py` | `oi_regime(closes, oi, volumes, period)` |
| OI Stress | `oi_stress.py` | `oi_stress(closes, oi, volumes, period)` |
| Price Inertia | `price_inertia.py` | `price_inertia(closes, period)` |
| Composite Regime | `regime_score.py` | `composite_regime(closes, highs, lows, period)` |
| Switch Speed | `regime_switch_speed.py` | `switch_speed(regime_labels, period)` |
| Runs Test | `runs_test.py` | `runs_test(closes, period)` |
| Sample Entropy | `sample_entropy.py` | `sample_entropy(data, m, r_mult, period)` |
| Dominant Cycle | `spectral_density.py` | `dominant_cycle(data, period)` |
| Stationarity | `stationarity_score.py` | `stationarity(closes, period)` |
| Structural Break | `structural_break.py` | `cusum_break(data, period, threshold)` |
| Tail Index | `tail_index.py` | `hill_tail_index(data, period, k_fraction)` |
| Trend Persistence | `trend_persistence.py` | `trend_persistence(data, max_lag, period)` |
| Trend Strength | `trend_strength_composite.py` | `trend_strength(closes, highs, lows, period)` |
| Turbulence | `turbulence_index.py` | `turbulence(returns_matrix, period)` |
| Variance Ratio | `variance_ratio.py` | `variance_ratio_test(data, period, holding)` |
| Vol Regime Threshold | `vol_regime_threshold.py` | `vol_regime_simple(closes, period)` |
| Vol Clustering | `volatility_clustering.py` | `vol_clustering(data, period)` |

## Spread（25 个）

| 指标 | 文件 | 函数 |
|------|------|------|
| Basis | `basis.py` | `basis(front_closes, back_closes, period)` |
| Rolling Beta | `beta.py` | `rolling_beta(asset_returns, benchmark_returns, period)` |
| Carry Signal | `carry.py` | `carry_signal(front_closes, back_closes, period)` |
| Cointegration | `cointegration_residual.py` | `cointegration_residual(closes_a, closes_b, period)` |
| Common Factor | `common_factor.py` | `common_factor(closes_list, period)` |
| Contagion | `contagion.py` | `contagion_score(returns_a, returns_b, period, threshold)` |
| Correlation Regime | `correlation_regime.py` | `correlation_regime(closes_a, closes_b, fast, slow)` |
| Cross-Asset RSI | `cross_asset_rsi.py` | `cross_asset_rsi(closes_a, closes_b, period)` |
| Cross Momentum | `cross_momentum.py` | `cross_momentum(closes_a, closes_b, period)` |
| Dispersion | `dispersion.py` | `dispersion(closes_list, period)` |
| Dynamic Hedge | `dynamic_hedge_ratio.py` | `dynamic_hedge(closes_a, closes_b, period)` |
| Energy-Metal Ratio | `energy_metal_ratio.py` | `energy_metal_ratio(energy_closes, metal_closes, period)` |
| Gold-Silver Ratio | `gold_silver_ratio.py` | `gold_silver_ratio(au_closes, ag_closes, period)` |
| Hedging Pressure | `hedging_pressure.py` | `hedging_pressure(closes, oi, volumes, period)` |
| Intermarket Divergence | `intermarket_divergence.py` | `intermarket_divergence(closes_a, closes_b, period)` |
| Lead-Lag | `lead_lag.py` | `lead_lag(closes_a, closes_b, max_lag, period)` |
| Metal Ratio | `metal_ratio.py` | `metal_ratio(closes_a, closes_b, period)` |
| Pair Z-Score | `pair_zscore.py` | `pair_zscore(closes_a, closes_b, period, method)` |
| Ratio Momentum | `ratio_momentum.py` | `ratio_momentum(closes_a, closes_b, period, lookback)` |
| Relative Strength | `relative_strength.py` | `relative_strength(asset_closes, benchmark_closes, period)` |
| Relative Value Z | `relative_value_zscore.py` | `rv_zscore(closes_a, closes_b, closes_c, period)` |
| Residual Momentum | `residual_momentum.py` | `residual_momentum(asset_closes, factor_closes, period, mom_period)` |
| Sector Momentum | `sector_momentum.py` | `sector_momentum(closes_list, period)` |
| Spread Volatility | `spread_volatility.py` | `spread_volatility(closes_a, closes_b, period)` |
| Term Premium | `term_premium.py` | `term_premium(front_closes, back_closes, period)` |

## Structure（24 个）

| 指标 | 文件 | 函数 |
|------|------|------|
| Commitment Ratio | `commitment_ratio.py` | `commitment_ratio(oi, volumes, period)` |
| Delivery Pressure | `delivery_pressure.py` | `delivery_pressure(oi, volumes, datetimes, period)` |
| Market Depth Proxy | `market_depth_proxy.py` | `depth_proxy(highs, lows, volumes, period)` |
| Net Positioning | `net_positioning.py` | `net_positioning_proxy(closes, oi, volumes, period)` |
| OI Bollinger | `oi_bollinger.py` | `oi_bollinger(oi, period, num_std)` |
| OI Breakout | `oi_breakout.py` | `oi_breakout(oi, period, threshold)` |
| OI Concentration | `oi_concentration.py` | `oi_concentration(oi, period)` |
| OI Divergence Enhanced | `oi_divergence_enhanced.py` | `oi_divergence_enhanced(closes, oi, volumes, period)` |
| OI Mean Reversion | `oi_mean_reversion.py` | `oi_mean_reversion(oi, period)` |
| OI Momentum Divergence | `oi_momentum_divergence.py` | `oi_momentum_price_divergence(closes, oi, period)` |
| OI Persistence | `oi_persistence.py` | `oi_persistence(oi, period)` |
| OI Price Regime | `oi_price_regime.py` | `oi_price_regime(closes, oi, period)` |
| OI Relative Strength | `oi_relative_strength.py` | `oi_relative_strength(oi, volumes, period)` |
| OI Velocity | `oi_velocity.py` | `oi_velocity(oi, period)` |
| OI-Volume Divergence | `oi_volume_divergence.py` | `oi_volume_divergence(oi, volumes, period)` |
| OI Weighted Price | `oi_weighted_price.py` | `oi_weighted_price(closes, oi, period)` |
| Position Crowding | `position_crowding.py` | `position_crowding(closes, oi, volumes, period)` |
| PVT Strength | `price_volume_trend_strength.py` | `pvt_strength(closes, volumes, period)` |
| Rollover Detector | `rollover_detector.py` | `rollover_detector(volumes, factors, ...)` / `post_rollover_momentum(...)` |
| Smart Money Index | `smart_money.py` | `smart_money_index(opens, closes, highs, lows, volumes, period)` |
| Speculation Index | `speculation_index.py` | `speculation_index(volumes, oi, period)` |
| Squeeze Detector | `squeeze_detector.py` | `squeeze_probability(closes, oi, volumes, period)` |
| Volume-OI Ratio | `volume_oi_ratio.py` | `volume_oi_ratio(volumes, oi, period)` |
| Warehouse Proxy | `warehouse_proxy.py` | `inventory_proxy(closes, oi, volumes, period)` |

## Microstructure（15 个）

| 指标 | 文件 | 函数 |
|------|------|------|
| Adverse Selection | `adverse_selection.py` | `adverse_selection(closes, volumes, period)` |
| Amihud Illiquidity | `amihud.py` | `amihud_illiquidity(closes, volumes, period)` |
| High-Low Spread | `high_low_spread.py` | `hl_spread(highs, lows, closes, period)` |
| Overnight Return | `overnight_return.py` | `overnight_return(opens, closes, period)` |
| Price Efficiency | `price_efficiency.py` | `price_efficiency_coefficient(closes, period)` |
| Kyle Lambda | `price_impact.py` | `kyle_lambda(closes, volumes, period)` |
| Range-to-Volume | `range_to_volume.py` | `range_to_volume(highs, lows, volumes, period)` |
| Realized Spread | `realized_spread_proxy.py` | `realized_spread(closes, period)` |
| Roll Spread | `roll_spread.py` | `roll_spread_estimate(closes, period)` |
| Tick Direction | `tick_direction.py` | `tick_direction(closes, period)` |
| Trade Clustering | `trade_clustering.py` | `trade_clustering(volumes, period)` |
| Trade Intensity | `trade_intensity.py` | `trade_intensity(volumes, period)` |
| Volume Clock | `volume_clock.py` | `volume_clock(volumes, target_volume, period)` |
| Volume Concentration | `volume_concentration.py` | `volume_concentration(volumes, period, top_pct)` |
| Volume Imbalance | `volume_imbalance.py` | `volume_imbalance(closes, volumes, period)` |

## Seasonality（15 个）

| 指标 | 文件 | 函数 |
|------|------|------|
| Contract Roll | `contract_roll.py` | `roll_effect(closes, is_rollover, period)` |
| Expiry Week | `expiry_week.py` | `expiry_week_effect(closes, datetimes, lookback)` |
| Holiday Proximity | `holiday_proximity.py` | `holiday_effect(closes, datetimes, lookback)` |
| Intraweek Pattern | `intraweek_pattern.py` | `intraweek_momentum(closes, datetimes, lookback)` |
| Month Cycle | `month_of_year.py` | `month_cycle(datetimes)` |
| Month Turn | `month_turn_effect.py` | `month_turn(closes, datetimes, window)` |
| Monthly Seasonal | `monthly_pattern.py` | `monthly_seasonal(closes, datetimes, lookback_years)` |
| Quarter Effect | `quarter_effect.py` | `quarter_end_effect(closes, datetimes, window)` |
| Seasonal Strength | `seasonal_decompose.py` | `seasonal_strength(closes, period)` |
| Seasonal Momentum | `seasonal_momentum.py` | `seasonal_momentum(closes, datetimes, lookback_years)` |
| Seasonal Z-Score | `seasonal_zscore.py` | `seasonal_zscore(closes, datetimes, period)` |
| Trading Day Number | `trading_day_number.py` | `trading_day_of_month(datetimes)` |
| Vol Seasonality | `volatility_seasonality.py` | `vol_seasonality(closes, datetimes, vol_period)` |
| Weekday Effect | `weekday_effect.py` | `weekday_effect(closes, datetimes, lookback)` |
| Year Cycle | `year_progress.py` | `year_cycle(datetimes)` |

---

## v1 → v2 变更记录

### 分类移动（8 个文件）
- `kama.py`: momentum → **trend**（自适应均线，不是动量）
- `acceleration_bands.py`: momentum → **volatility**（波动率通道）
- `chop_zone.py`: momentum → **regime**（趋势/震荡分类）
- `trix.py`: trend → **momentum**（ROC of triple EMA）
- `intraday_intensity.py`: volatility → **volume**（量价指标）
- `price_acceleration.py`: volatility → **momentum**（动量二阶导）
- `close_location.py`: volume → **momentum**（纯价格指标）
- `price_density.py`: volume → **volatility**（波动率/结构指标）

### ML → Volatility（5 个纯统计指标）
- `information_ratio.py`, `robust_zscore.py`, `quantile_bands.py`, `rolling_percentile_rank.py`, `successive_differences.py`

### Spread → Momentum（1 个单品种指标）
- `sma_ratio_zscore.py`

### 重命名（3 个文件）
- `chop.py` → `atr_ratio.py`（函数名一致）
- `chop_zone.py` → `choppiness_index.py`（函数名一致）
- `vol_regime_markov.py` → `vol_regime_threshold.py`（实现不是 Markov）

### 代码改进
- 新增 `_utils.py` 共享 helper（_ema, _ema_skip_nan, _sma, _rsi），消除 16 个文件的重复
- `decycler.py` 移除死代码
- `correlation_breakdown.py` 修复参数名（returns_a → closes_a）
- Seasonality 全部 15 个文件添加 module docstring
