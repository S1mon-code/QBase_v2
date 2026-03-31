"""Tests for the monitoring package.

Covers:
- decay_detector: rolling sharpe yellow/red/clear, backtest deviation, trade frequency
- regime_alert: each mismatch type, no mismatch returns None
- retirement: uses portfolio.retirement correctly, edge cases
- dashboard: basic structure, empty data, with alerts
- run_all_checks: multiple alerts, no alerts, partial data
"""

from __future__ import annotations

import numpy as np
import pytest

from monitoring.decay_detector import (
    DecayAlert,
    check_backtest_deviation,
    check_rolling_sharpe,
    check_trade_frequency,
    run_all_checks,
)
from monitoring.regime_alert import RegimeAlert, check_regime_consistency
from monitoring.retirement import (
    _consecutive_loss_months,
    _rolling_sharpe,
    monitor_strategy_health,
)
from monitoring.dashboard import (
    DashboardSummary,
    StrategyStatus,
    generate_dashboard,
)


# ═══════════════════════════════════════════════════════════════════════════
# decay_detector — check_rolling_sharpe
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckRollingSharpe:
    """Tests for rolling Sharpe decay detection."""

    def test_clear_when_positive_returns(self) -> None:
        """Consistently positive returns should produce no alert."""
        returns = np.full(120, 0.001)  # +0.1% daily
        assert check_rolling_sharpe(returns) is None

    def test_yellow_alert(self) -> None:
        """Sustained negative returns triggers yellow."""
        # Varied negative returns so std > 0 and mean < 0 => Sharpe < 0
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.005, 0.002, 100)
        alert = check_rolling_sharpe(returns, window=20, yellow_threshold=10, red_threshold=999)
        assert alert is not None
        assert alert.level == "yellow"
        assert alert.metric == "rolling_sharpe"

    def test_red_alert(self) -> None:
        """Sustained negative returns exceeding red threshold."""
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.005, 0.002, 100)
        alert = check_rolling_sharpe(returns, window=20, yellow_threshold=10, red_threshold=20)
        assert alert is not None
        assert alert.level == "red"

    def test_insufficient_data_returns_none(self) -> None:
        """Returns None when data is shorter than window."""
        returns = np.full(30, -0.005)
        assert check_rolling_sharpe(returns, window=60) is None

    def test_all_zero_returns(self) -> None:
        """All-zero returns should not trigger (Sharpe == 0, not < 0)."""
        returns = np.zeros(120)
        assert check_rolling_sharpe(returns) is None


# ═══════════════════════════════════════════════════════════════════════════
# decay_detector — check_backtest_deviation
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckBacktestDeviation:
    """Tests for backtest deviation check."""

    def test_no_alert_within_range(self) -> None:
        """No alert when live Sharpe is close to backtest."""
        assert check_backtest_deviation(1.0, 1.2, 0.3) is None

    def test_yellow_alert(self) -> None:
        """Yellow alert when deviation is between yellow and red."""
        # deviation = (1.5 - 0.8) / 0.4 = 1.75 -> yellow (>1.5, <2.0)
        alert = check_backtest_deviation(0.8, 1.5, 0.4)
        assert alert is not None
        assert alert.level == "yellow"
        assert alert.metric == "backtest_deviation"

    def test_red_alert(self) -> None:
        """Red alert when deviation exceeds red threshold."""
        # deviation = (2.0 - 0.5) / 0.5 = 3.0 -> red
        alert = check_backtest_deviation(0.5, 2.0, 0.5)
        assert alert is not None
        assert alert.level == "red"

    def test_zero_std_returns_none(self) -> None:
        """Returns None when backtest_std is 0."""
        assert check_backtest_deviation(0.5, 1.0, 0.0) is None

    def test_negative_std_returns_none(self) -> None:
        """Returns None when backtest_std is negative."""
        assert check_backtest_deviation(0.5, 1.0, -0.1) is None

    def test_live_better_than_backtest(self) -> None:
        """No alert when live Sharpe exceeds backtest."""
        assert check_backtest_deviation(2.0, 1.0, 0.3) is None


# ═══════════════════════════════════════════════════════════════════════════
# decay_detector — check_trade_frequency
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckTradeFrequency:
    """Tests for trade frequency deviation."""

    def test_no_alert_within_range(self) -> None:
        """No alert when frequency is similar."""
        assert check_trade_frequency(10.0, 12.0) is None

    def test_yellow_alert_high(self) -> None:
        """Yellow alert when frequency is 60% higher than expected."""
        # deviation = |16 - 10| / 10 = 0.6 -> yellow (>0.5, <1.0)
        alert = check_trade_frequency(16.0, 10.0)
        assert alert is not None
        assert alert.level == "yellow"

    def test_yellow_alert_low(self) -> None:
        """Yellow alert when frequency is 60% lower than expected."""
        alert = check_trade_frequency(4.0, 10.0)
        assert alert is not None
        assert alert.level == "yellow"

    def test_red_alert(self) -> None:
        """Red alert when frequency deviates > 100%."""
        alert = check_trade_frequency(25.0, 10.0)
        assert alert is not None
        assert alert.level == "red"

    def test_zero_expected_returns_none(self) -> None:
        """Returns None when expected trades is 0."""
        assert check_trade_frequency(5.0, 0.0) is None


# ═══════════════════════════════════════════════════════════════════════════
# decay_detector — run_all_checks
# ═══════════════════════════════════════════════════════════════════════════

class TestRunAllChecks:
    """Tests for the aggregated run_all_checks."""

    def test_no_data_returns_empty(self) -> None:
        """No data provided -> no alerts."""
        assert run_all_checks() == []

    def test_partial_data_runs_available_checks(self) -> None:
        """Providing only some data runs only those checks."""
        # Provide only trade frequency data that triggers alert
        alerts = run_all_checks(actual_trades=25.0, expected_trades=10.0)
        assert len(alerts) == 1
        assert alerts[0].metric == "trade_frequency"

    def test_multiple_alerts(self) -> None:
        """Multiple failing checks produce multiple alerts."""
        # Bad returns + bad trade frequency
        bad_returns = np.full(120, -0.005)
        alerts = run_all_checks(
            daily_returns=bad_returns,
            actual_trades=25.0,
            expected_trades=10.0,
        )
        assert len(alerts) >= 2

    def test_all_clear(self) -> None:
        """Good data produces no alerts."""
        good_returns = np.full(120, 0.001)
        alerts = run_all_checks(
            daily_returns=good_returns,
            live_sharpe=1.0,
            backtest_sharpe=1.1,
            backtest_std=0.3,
            actual_trades=10.0,
            expected_trades=11.0,
        )
        assert alerts == []


# ═══════════════════════════════════════════════════════════════════════════
# regime_alert
# ═══════════════════════════════════════════════════════════════════════════

class TestRegimeAlert:
    """Tests for regime consistency checks."""

    def test_no_mismatch_mild_trend(self) -> None:
        """Normal mild_trend conditions produce no alert."""
        assert check_regime_consistency("mild_trend", 50, 25, 3.0) is None

    def test_no_mismatch_strong_trend(self) -> None:
        """Normal strong_trend conditions produce no alert."""
        assert check_regime_consistency("strong_trend", 60, 30, 8.0) is None

    def test_mild_trend_extreme_vol(self) -> None:
        """Mild trend with extreme volatility -> critical."""
        alert = check_regime_consistency("mild_trend", 95, 25, 3.0)
        assert alert is not None
        assert alert.severity == "critical"
        assert alert.detected_behavior == "extreme_volatility"

    def test_mean_reversion_extreme_vol(self) -> None:
        """Mean reversion with extreme volatility -> critical."""
        alert = check_regime_consistency("mean_reversion", 95, 25, 3.0)
        assert alert is not None
        assert alert.severity == "critical"

    def test_mean_reversion_strong_move(self) -> None:
        """Mean reversion with big return -> warning."""
        alert = check_regime_consistency("mean_reversion", 50, 25, 18.0)
        assert alert is not None
        assert alert.severity == "warning"
        assert alert.detected_behavior == "strong_directional_move"

    def test_strong_trend_weak_adx(self) -> None:
        """Strong trend but ADX < 15 -> warning."""
        alert = check_regime_consistency("strong_trend", 50, 10, 5.0)
        assert alert is not None
        assert alert.severity == "warning"
        assert alert.detected_behavior == "weak_trend"

    def test_mild_trend_upgrade(self) -> None:
        """Mild trend with strong ADX + large return -> info."""
        alert = check_regime_consistency("mild_trend", 50, 45, 12.0)
        assert alert is not None
        assert alert.severity == "info"
        assert alert.detected_behavior == "possible_strong_trend"

    def test_crisis_calm_market(self) -> None:
        """Crisis assigned but calm market -> warning."""
        alert = check_regime_consistency("crisis", 20, 15, 1.0)
        assert alert is not None
        assert alert.severity == "warning"
        assert alert.detected_behavior == "calm_market"

    def test_no_mismatch_mean_reversion(self) -> None:
        """Normal mean reversion conditions produce no alert."""
        assert check_regime_consistency("mean_reversion", 50, 20, 2.0) is None


# ═══════════════════════════════════════════════════════════════════════════
# retirement monitor
# ═══════════════════════════════════════════════════════════════════════════

class TestRetirementMonitor:
    """Tests for monitoring.retirement."""

    def test_normal_strategy(self) -> None:
        """Healthy strategy returns normal."""
        monthly = np.array([0.02, 0.01, 0.03, 0.02, 0.01, 0.02,
                            0.03, 0.01, 0.02, 0.01, 0.02, 0.03])
        result = monitor_strategy_health("strat_a", monthly, -0.05, -0.10)
        assert result.action == "normal"
        assert result.strategy == "strat_a"

    def test_observe_on_negative_6m_sharpe(self) -> None:
        """Negative 6m Sharpe triggers observe."""
        # Last 6 months negative
        monthly = np.array([0.02, 0.01, 0.03, 0.02, 0.01, 0.02,
                            -0.03, -0.02, -0.01, -0.02, -0.03, -0.02])
        result = monitor_strategy_health("strat_b", monthly, -0.05, -0.10)
        assert result.action == "observe"

    def test_observe_on_consecutive_losses(self) -> None:
        """3+ consecutive loss months triggers observe."""
        monthly = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                            0.05, 0.05, 0.05, -0.01, -0.01, -0.01])
        result = monitor_strategy_health("strat_c", monthly, -0.02, -0.10)
        # May be observe due to consecutive losses or 6m sharpe
        assert result.action in ("observe", "remove")

    def test_immediate_remove_on_dd(self) -> None:
        """DD exceeding 1.5x backtest max triggers immediate removal."""
        monthly = np.full(12, 0.01)
        result = monitor_strategy_health("strat_d", monthly, -0.20, -0.10)
        assert result.action == "immediate_remove"

    def test_remove_on_bad_12m_sharpe(self) -> None:
        """Very negative 12m Sharpe triggers remove."""
        monthly = np.full(12, -0.05)
        result = monitor_strategy_health("strat_e", monthly, -0.05, -0.10)
        assert result.action == "remove"

    def test_short_history(self) -> None:
        """Works with fewer than 12 months of data."""
        monthly = np.array([0.02, 0.01, 0.03])
        result = monitor_strategy_health("strat_f", monthly, -0.02, -0.10)
        assert result.strategy == "strat_f"
        assert result.action in ("normal", "observe", "remove", "immediate_remove")


# ═══════════════════════════════════════════════════════════════════════════
# retirement helper functions
# ═══════════════════════════════════════════════════════════════════════════

class TestRetirementHelpers:
    """Tests for internal helper functions."""

    def test_consecutive_loss_months_all_positive(self) -> None:
        """No losses at tail returns 0."""
        assert _consecutive_loss_months(np.array([0.01, 0.02, 0.01])) == 0

    def test_consecutive_loss_months_all_negative(self) -> None:
        """All negative returns full length."""
        assert _consecutive_loss_months(np.array([-0.01, -0.02, -0.03])) == 3

    def test_consecutive_loss_months_mixed(self) -> None:
        """Only counts from the tail."""
        assert _consecutive_loss_months(np.array([-0.01, 0.02, -0.01, -0.02])) == 2

    def test_rolling_sharpe_all_positive(self) -> None:
        """Positive returns produce positive Sharpe."""
        monthly = np.array([0.02, 0.03, 0.01, 0.02, 0.03, 0.01])
        assert _rolling_sharpe(monthly, 6) > 0

    def test_rolling_sharpe_zero_std(self) -> None:
        """Identical returns produce 0 Sharpe (no risk signal)."""
        monthly = np.full(6, 0.02)
        assert _rolling_sharpe(monthly, 6) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# dashboard
# ═══════════════════════════════════════════════════════════════════════════

class TestDashboard:
    """Tests for the monitoring dashboard."""

    def test_empty_dashboard(self) -> None:
        """No strategy data produces empty dashboard."""
        dash = generate_dashboard("RB")
        assert dash.instrument == "RB"
        assert dash.n_active_strategies == 0
        assert dash.strategies == ()
        assert dash.alerts == ()

    def test_basic_structure(self) -> None:
        """Dashboard with strategy data returns correct structure."""
        returns = {"strat_1": np.full(120, 0.001)}
        dash = generate_dashboard("RB", strategy_returns=returns, regime="mild_trend")
        assert dash.instrument == "RB"
        assert dash.active_regime == "mild_trend"
        assert dash.n_active_strategies == 1
        assert len(dash.strategies) == 1
        assert dash.strategies[0].name == "strat_1"

    def test_with_alerts(self) -> None:
        """Bad returns generate alerts in the dashboard."""
        bad_returns = np.full(120, -0.005)
        returns = {"bad_strat": bad_returns}
        dash = generate_dashboard("RB", strategy_returns=returns)
        # Should have at least one alert from rolling sharpe
        assert len(dash.alerts) > 0
        assert "[bad_strat]" in dash.alerts[0]

    def test_dashboard_with_multiple_strategies(self) -> None:
        """Multiple strategies are all represented."""
        returns = {
            "strat_a": np.full(120, 0.001),
            "strat_b": np.full(120, 0.002),
        }
        dash = generate_dashboard("HC", strategy_returns=returns)
        assert dash.n_active_strategies == 2
        names = {s.name for s in dash.strategies}
        assert names == {"strat_a", "strat_b"}

    def test_dashboard_sharpe_calculation(self) -> None:
        """60d Sharpe is computed when sufficient data exists."""
        # Use varied positive returns so std > 0
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.01, 120)
        returns = {"strat_1": rets}
        dash = generate_dashboard("RB", strategy_returns=returns)
        assert dash.strategies[0].current_sharpe_60d is not None

    def test_dashboard_insufficient_data_sharpe(self) -> None:
        """60d Sharpe is None when insufficient data."""
        returns = {"strat_1": np.full(30, 0.001)}
        dash = generate_dashboard("RB", strategy_returns=returns)
        assert dash.strategies[0].current_sharpe_60d is None

    def test_dashboard_frozen_dataclasses(self) -> None:
        """StrategyStatus and DashboardSummary are immutable."""
        dash = generate_dashboard("RB")
        with pytest.raises(AttributeError):
            dash.instrument = "HC"  # type: ignore[misc]

    def test_dashboard_stop_level_passthrough(self) -> None:
        """Stop level is passed through correctly."""
        dash = generate_dashboard("RB", stop_level="reduce")
        assert dash.stop_level == "reduce"

    def test_dashboard_portfolio_dd_passthrough(self) -> None:
        """Portfolio drawdown is passed through correctly."""
        dash = generate_dashboard("RB", portfolio_dd=-0.12)
        assert dash.portfolio_dd == -0.12
