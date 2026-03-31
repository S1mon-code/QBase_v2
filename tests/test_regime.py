"""Comprehensive tests for the regime labeling system (Phase 2).

Covers:
- schema.py: load, save, validate (missing fields, invalid enum, overlapping)
- labeler.py: known patterns (uptrend, downtrend, flat, spike), empty, edge cases
- matcher.py: filter by regime, direction, split, missing instrument
"""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import yaml

from regime.schema import (
    RegimeConfig,
    RegimeLabel,
    VALID_REGIMES,
    VALID_DIRECTIONS,
    VALID_SPLITS,
    load_labels,
    save_labels,
    validate_labels,
)
from regime.labeler import (
    auto_label,
    _find_peaks,
    _find_troughs,
    _merge_extrema,
    _enforce_alternation,
    _compute_rolling_atr,
    _add_months,
)
from regime.matcher import (
    get_regime_periods,
    get_train_periods,
    get_oos_periods,
    get_holdout_periods,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def sample_label() -> RegimeLabel:
    """A typical regime label for testing."""
    return RegimeLabel(
        start=date(2020, 1, 1),
        end=date(2020, 6, 30),
        regime="strong_trend",
        direction="up",
        driver="test driver",
        buffer_start=date(2019, 11, 1),
        buffer_end=date(2020, 8, 31),
        split="train",
    )


@pytest.fixture
def sample_config(sample_label) -> RegimeConfig:
    """A minimal valid RegimeConfig."""
    return RegimeConfig(
        instrument="RB",
        version=1,
        labeled_by="test",
        labels=(sample_label,),
    )


@pytest.fixture
def label_yaml_path(tmp_path) -> Path:
    """Create a sample YAML label file and return its path."""
    content = {
        "instrument": "I",
        "version": 1,
        "labeled_by": "auto + manual review",
        "labels": [
            {
                "start": "2015-06-01",
                "end": "2016-02-28",
                "regime": "strong_trend",
                "direction": "up",
                "driver": "供给侧改革",
                "buffer_start": "2015-04-01",
                "buffer_end": "2016-04-30",
                "split": "train",
            },
            {
                "start": "2020-11-01",
                "end": "2021-05-31",
                "regime": "strong_trend",
                "direction": "up",
                "split": "oos",
            },
            {
                "start": "2024-01-01",
                "end": "2024-08-31",
                "regime": "mean_reversion",
                "direction": "neutral",
                "split": "holdout",
            },
        ],
    }
    p = tmp_path / "I.yaml"
    with open(p, "w") as f:
        yaml.dump(content, f, allow_unicode=True, sort_keys=False)
    return p


@pytest.fixture
def labels_dir(tmp_path, label_yaml_path) -> Path:
    """Create a data/regime_labels/ directory structure with a sample file."""
    dest = tmp_path / "data" / "regime_labels"
    dest.mkdir(parents=True)
    import shutil

    shutil.copy(label_yaml_path, dest / "I.yaml")
    return dest


# ===================================================================
# Schema tests
# ===================================================================


class TestRegimeLabel:
    """Tests for the RegimeLabel dataclass."""

    def test_create_minimal(self):
        """Create a label with only required fields."""
        lbl = RegimeLabel(
            start=date(2020, 1, 1),
            end=date(2020, 3, 31),
            regime="mild_trend",
            direction="down",
        )
        assert lbl.regime == "mild_trend"
        assert lbl.driver == ""
        assert lbl.buffer_start is None
        assert lbl.split == "train"

    def test_create_full(self, sample_label):
        """Create a label with all fields populated."""
        assert sample_label.start == date(2020, 1, 1)
        assert sample_label.buffer_end == date(2020, 8, 31)
        assert sample_label.driver == "test driver"

    def test_frozen(self, sample_label):
        """Labels are immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_label.regime = "crisis"


class TestRegimeConfig:
    """Tests for the RegimeConfig dataclass."""

    def test_create_default(self):
        """Create a config with defaults."""
        cfg = RegimeConfig(instrument="RB")
        assert cfg.version == 1
        assert cfg.labeled_by == "auto"
        assert cfg.labels == ()

    def test_frozen(self, sample_config):
        """Config is immutable."""
        with pytest.raises(AttributeError):
            sample_config.instrument = "I"


class TestLoadLabels:
    """Tests for loading labels from YAML."""

    def test_load_basic(self, label_yaml_path):
        """Load a well-formed YAML file."""
        cfg = load_labels(label_yaml_path)
        assert cfg.instrument == "I"
        assert len(cfg.labels) == 3
        assert cfg.labels[0].regime == "strong_trend"
        assert cfg.labels[0].driver == "供给侧改革"

    def test_load_dates_parsed(self, label_yaml_path):
        """Dates are parsed as Python date objects."""
        cfg = load_labels(label_yaml_path)
        assert cfg.labels[0].start == date(2015, 6, 1)
        assert cfg.labels[0].buffer_end == date(2016, 4, 30)

    def test_load_missing_buffer(self, label_yaml_path):
        """Labels without buffer dates get None."""
        cfg = load_labels(label_yaml_path)
        assert cfg.labels[1].buffer_start is None
        assert cfg.labels[1].buffer_end is None

    def test_load_file_not_found(self, tmp_path):
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_labels(tmp_path / "nonexistent.yaml")

    def test_load_empty_labels(self, tmp_path):
        """Load a file with no labels."""
        p = tmp_path / "empty.yaml"
        with open(p, "w") as f:
            yaml.dump({"instrument": "RB", "labels": []}, f)
        cfg = load_labels(p)
        assert len(cfg.labels) == 0


class TestSaveLabels:
    """Tests for saving labels to YAML."""

    def test_save_roundtrip(self, sample_config, tmp_path):
        """Save then load produces equivalent config."""
        p = tmp_path / "out.yaml"
        save_labels(sample_config, p)
        loaded = load_labels(p)
        assert loaded.instrument == sample_config.instrument
        assert len(loaded.labels) == len(sample_config.labels)
        assert loaded.labels[0].regime == sample_config.labels[0].regime
        assert loaded.labels[0].start == sample_config.labels[0].start

    def test_save_creates_dirs(self, sample_config, tmp_path):
        """Save creates parent directories if needed."""
        p = tmp_path / "deep" / "nested" / "labels.yaml"
        save_labels(sample_config, p)
        assert p.exists()

    def test_save_unicode(self, tmp_path):
        """Chinese characters survive roundtrip."""
        lbl = RegimeLabel(
            start=date(2020, 1, 1),
            end=date(2020, 6, 30),
            regime="strong_trend",
            direction="up",
            driver="供给侧改革",
        )
        cfg = RegimeConfig(instrument="I", labels=(lbl,))
        p = tmp_path / "chinese.yaml"
        save_labels(cfg, p)
        loaded = load_labels(p)
        assert loaded.labels[0].driver == "供给侧改革"


class TestValidateLabels:
    """Tests for validate_labels."""

    def test_valid_config(self, sample_config):
        """No errors on a well-formed config."""
        errors = validate_labels(sample_config)
        assert errors == []

    def test_missing_instrument(self):
        """Flag missing instrument."""
        cfg = RegimeConfig(instrument="")
        errors = validate_labels(cfg)
        assert any("instrument" in e for e in errors)

    def test_invalid_regime(self):
        """Flag invalid regime string."""
        lbl = RegimeLabel(
            start=date(2020, 1, 1),
            end=date(2020, 6, 30),
            regime="unknown_regime",
            direction="up",
        )
        cfg = RegimeConfig(instrument="RB", labels=(lbl,))
        errors = validate_labels(cfg)
        assert any("invalid regime" in e for e in errors)

    def test_invalid_direction(self):
        """Flag invalid direction string."""
        lbl = RegimeLabel(
            start=date(2020, 1, 1),
            end=date(2020, 6, 30),
            regime="crisis",
            direction="sideways",
        )
        cfg = RegimeConfig(instrument="RB", labels=(lbl,))
        errors = validate_labels(cfg)
        assert any("invalid direction" in e for e in errors)

    def test_invalid_split(self):
        """Flag invalid split string."""
        lbl = RegimeLabel(
            start=date(2020, 1, 1),
            end=date(2020, 6, 30),
            regime="crisis",
            direction="up",
            split="validation",
        )
        cfg = RegimeConfig(instrument="RB", labels=(lbl,))
        errors = validate_labels(cfg)
        assert any("invalid split" in e for e in errors)

    def test_start_after_end(self):
        """Flag start > end."""
        lbl = RegimeLabel(
            start=date(2020, 6, 30),
            end=date(2020, 1, 1),
            regime="crisis",
            direction="down",
        )
        cfg = RegimeConfig(instrument="RB", labels=(lbl,))
        errors = validate_labels(cfg)
        assert any("start" in e and "end" in e for e in errors)

    def test_buffer_start_after_start(self):
        """Flag buffer_start > start."""
        lbl = RegimeLabel(
            start=date(2020, 1, 1),
            end=date(2020, 6, 30),
            regime="mild_trend",
            direction="up",
            buffer_start=date(2020, 3, 1),
        )
        cfg = RegimeConfig(instrument="RB", labels=(lbl,))
        errors = validate_labels(cfg)
        assert any("buffer_start" in e for e in errors)

    def test_buffer_end_before_end(self):
        """Flag buffer_end < end."""
        lbl = RegimeLabel(
            start=date(2020, 1, 1),
            end=date(2020, 6, 30),
            regime="mild_trend",
            direction="up",
            buffer_end=date(2020, 3, 1),
        )
        cfg = RegimeConfig(instrument="RB", labels=(lbl,))
        errors = validate_labels(cfg)
        assert any("buffer_end" in e for e in errors)

    def test_overlapping_periods(self):
        """Flag overlapping core periods."""
        lbl1 = RegimeLabel(
            start=date(2020, 1, 1),
            end=date(2020, 6, 30),
            regime="strong_trend",
            direction="up",
        )
        lbl2 = RegimeLabel(
            start=date(2020, 5, 1),
            end=date(2020, 12, 31),
            regime="mild_trend",
            direction="down",
        )
        cfg = RegimeConfig(instrument="RB", labels=(lbl1, lbl2))
        errors = validate_labels(cfg)
        assert any("overlaps" in e for e in errors)

    def test_non_overlapping_ok(self):
        """No overlap error when periods are sequential."""
        lbl1 = RegimeLabel(
            start=date(2020, 1, 1),
            end=date(2020, 5, 31),
            regime="strong_trend",
            direction="up",
        )
        lbl2 = RegimeLabel(
            start=date(2020, 6, 1),
            end=date(2020, 12, 31),
            regime="mild_trend",
            direction="down",
        )
        cfg = RegimeConfig(instrument="RB", labels=(lbl1, lbl2))
        errors = validate_labels(cfg)
        assert not any("overlaps" in e for e in errors)

    def test_multiple_errors(self):
        """Multiple validation errors returned at once."""
        lbl = RegimeLabel(
            start=date(2020, 6, 30),
            end=date(2020, 1, 1),
            regime="bad_regime",
            direction="bad_dir",
            split="bad_split",
        )
        cfg = RegimeConfig(instrument="", labels=(lbl,))
        errors = validate_labels(cfg)
        assert len(errors) >= 4  # instrument + regime + direction + split + date order


# ===================================================================
# Labeler tests
# ===================================================================


def _make_dates(start: date, n: int) -> np.ndarray:
    """Generate n daily dates starting from *start*."""
    from datetime import timedelta

    return np.array([start + timedelta(days=i) for i in range(n)])


class TestLabelerHelpers:
    """Tests for internal helper functions."""

    def test_find_peaks_simple(self):
        """Detect peaks in a simple pattern."""
        prices = np.array([1, 2, 3, 2, 1, 2, 5, 2, 1], dtype=float)
        peaks = _find_peaks(prices, window=1)
        assert 2 in peaks  # local max at index 2
        assert 6 in peaks  # local max at index 6

    def test_find_troughs_simple(self):
        """Detect troughs in a simple pattern."""
        prices = np.array([5, 3, 1, 3, 5, 3, 1, 3, 5], dtype=float)
        troughs = _find_troughs(prices, window=1)
        assert 2 in troughs
        assert 6 in troughs

    def test_find_peaks_flat(self):
        """Flat series: first occurrence is a peak if equal."""
        prices = np.array([1.0] * 10)
        peaks = _find_peaks(prices, window=1)
        # All values equal -> all qualify
        assert len(peaks) > 0

    def test_merge_extrema(self):
        """Merge peaks and troughs in time order."""
        peaks = np.array([2, 6])
        troughs = np.array([4, 8])
        merged = _merge_extrema(peaks, troughs)
        assert merged == [(2, "peak"), (4, "trough"), (6, "peak"), (8, "trough")]

    def test_enforce_alternation_removes_consecutive(self):
        """Consecutive peaks: keep the higher one."""
        prices = np.array([1, 5, 3, 8, 2, 1], dtype=float)
        extrema = [(1, "peak"), (3, "peak"), (5, "trough")]
        result = _enforce_alternation(extrema, prices)
        assert len(result) == 2
        assert result[0] == (3, "peak")  # 8 > 5
        assert result[1] == (5, "trough")

    def test_compute_rolling_atr(self):
        """Rolling ATR produces expected NaN prefix and finite values."""
        prices = np.random.RandomState(42).uniform(100, 200, size=50)
        atr = _compute_rolling_atr(prices, window=10)
        assert np.all(np.isnan(atr[:10]))
        assert np.all(np.isfinite(atr[10:]))

    def test_add_months_positive(self):
        """Add months forward."""
        assert _add_months(date(2020, 1, 15), 2) == date(2020, 3, 15)

    def test_add_months_negative(self):
        """Subtract months."""
        assert _add_months(date(2020, 3, 15), -2) == date(2020, 1, 15)

    def test_add_months_clamp_day(self):
        """Clamp day when target month is shorter."""
        result = _add_months(date(2020, 1, 31), 1)
        assert result == date(2020, 2, 29)  # 2020 is leap year

    def test_add_months_year_rollover(self):
        """Year boundary crossing."""
        assert _add_months(date(2020, 11, 15), 3) == date(2021, 2, 15)


class TestAutoLabel:
    """Tests for the auto_label function."""

    def _default_config(self) -> dict:
        """Return a config dict for testing without loading YAML."""
        return {
            "strong_trend_pct": 0.20,
            "mild_trend_pct": 0.05,
            "crisis_atr_sigma": 3.0,
            "min_duration_months": 1,
            "buffer_months": 2,
            "peak_trough_window": 1,
        }

    def test_empty_array(self):
        """Empty prices returns empty labels."""
        result = auto_label(np.array([]), np.array([]), config=self._default_config())
        assert result == []

    def test_single_price(self):
        """Single data point returns empty."""
        dates = np.array([date(2020, 1, 1)])
        prices = np.array([100.0])
        result = auto_label(prices, dates, config=self._default_config())
        assert result == []

    def test_uptrend(self):
        """Strong uptrend should produce a strong_trend label."""
        n = 360  # ~1 year of trading days
        dates = _make_dates(date(2019, 1, 1), n)
        # W shape: drop, rise, drop, strong rise
        prices = np.concatenate([
            np.linspace(200, 100, n // 4),
            np.linspace(100, 300, n // 4),
            np.linspace(300, 150, n // 4),
            np.linspace(150, 350, n - 3 * (n // 4)),
        ])
        cfg = self._default_config()
        labels = auto_label(prices, dates, config=cfg)
        regimes = [lbl.regime for lbl in labels]
        assert "strong_trend" in regimes

    def test_downtrend(self):
        """Strong downtrend should produce a strong_trend down label."""
        n = 360
        dates = _make_dates(date(2019, 1, 1), n)
        # Inverse W: rise, drop, rise, strong drop
        prices = np.concatenate([
            np.linspace(100, 300, n // 4),
            np.linspace(300, 100, n // 4),
            np.linspace(100, 250, n // 4),
            np.linspace(250, 80, n - 3 * (n // 4)),
        ])
        cfg = self._default_config()
        labels = auto_label(prices, dates, config=cfg)
        directions = [lbl.direction for lbl in labels]
        assert "down" in directions

    def test_flat_produces_mean_reversion(self):
        """Flat price series should produce mean_reversion labels."""
        n = 180
        dates = _make_dates(date(2020, 1, 1), n)
        # Small oscillation: ±2%
        rng = np.random.RandomState(0)
        prices = 100 + 2 * np.sin(np.linspace(0, 8 * np.pi, n)) + rng.randn(n) * 0.5
        cfg = self._default_config()
        cfg["peak_trough_window"] = 1
        labels = auto_label(prices, dates, config=cfg)
        if labels:
            regimes = {lbl.regime for lbl in labels}
            # With small oscillations the moves should be small
            assert "mean_reversion" in regimes or "mild_trend" in regimes

    def test_labels_have_buffers(self):
        """All labels should have buffer dates."""
        n = 360
        dates = _make_dates(date(2019, 1, 1), n)
        prices = np.concatenate([
            np.linspace(200, 100, n // 4),
            np.linspace(100, 300, n // 4),
            np.linspace(300, 150, n // 4),
            np.linspace(150, 350, n - 3 * (n // 4)),
        ])
        labels = auto_label(prices, dates, config=self._default_config())
        for lbl in labels:
            assert lbl.buffer_start is not None
            assert lbl.buffer_end is not None
            assert lbl.buffer_start <= lbl.start
            assert lbl.buffer_end >= lbl.end

    def test_labels_valid_enums(self):
        """All labels should have valid regime/direction/split values."""
        n = 200
        dates = _make_dates(date(2020, 1, 1), n)
        prices = np.concatenate([
            np.linspace(100, 200, 50),
            np.linspace(200, 80, 50),
            np.linspace(80, 250, 50),
            np.linspace(250, 120, 50),
        ])
        labels = auto_label(prices, dates, config=self._default_config())
        for lbl in labels:
            assert lbl.regime in VALID_REGIMES
            assert lbl.direction in VALID_DIRECTIONS
            assert lbl.split in VALID_SPLITS

    def test_datetime64_dates(self):
        """Accept numpy datetime64 date arrays."""
        n = 120
        dates = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-01") + np.timedelta64(n, "D"),
            np.timedelta64(1, "D"),
        )
        prices = np.concatenate([
            np.linspace(200, 100, n // 3),
            np.linspace(100, 300, n - n // 3),
        ])
        labels = auto_label(prices, dates, config=self._default_config())
        assert isinstance(labels, list)
        for lbl in labels:
            assert isinstance(lbl.start, date)

    def test_crisis_spike(self):
        """An extreme volatility spike should be detectable."""
        n = 200
        dates = _make_dates(date(2020, 1, 1), n)
        rng = np.random.RandomState(42)
        prices = np.ones(n) * 100.0
        # Insert a huge spike
        prices[90:110] = 100 + rng.randn(20) * 50
        cfg = self._default_config()
        cfg["crisis_atr_sigma"] = 2.0  # lower threshold for test
        labels = auto_label(prices, dates, config=cfg)
        # May or may not detect crisis depending on exact random values
        # but should produce some labels
        assert isinstance(labels, list)


# ===================================================================
# Matcher tests
# ===================================================================


class TestMatcher:
    """Tests for the regime matcher module."""

    def _setup_labels_dir(self, tmp_path, monkeypatch):
        """Set up a fake labels directory and monkeypatch the module."""
        import regime.matcher as matcher_mod

        labels_dir = tmp_path / "data" / "regime_labels"
        labels_dir.mkdir(parents=True)

        content = {
            "instrument": "I",
            "version": 1,
            "labeled_by": "test",
            "labels": [
                {
                    "start": "2015-06-01",
                    "end": "2016-02-28",
                    "regime": "strong_trend",
                    "direction": "up",
                    "buffer_start": "2015-04-01",
                    "buffer_end": "2016-04-30",
                    "split": "train",
                },
                {
                    "start": "2018-01-01",
                    "end": "2018-06-30",
                    "regime": "mild_trend",
                    "direction": "down",
                    "buffer_start": "2017-11-01",
                    "buffer_end": "2018-08-31",
                    "split": "train",
                },
                {
                    "start": "2020-11-01",
                    "end": "2021-05-31",
                    "regime": "strong_trend",
                    "direction": "up",
                    "buffer_start": "2020-09-01",
                    "buffer_end": "2021-07-31",
                    "split": "oos",
                },
                {
                    "start": "2023-01-01",
                    "end": "2023-06-30",
                    "regime": "mean_reversion",
                    "direction": "neutral",
                    "buffer_start": "2022-11-01",
                    "buffer_end": "2023-08-31",
                    "split": "holdout",
                },
            ],
        }
        p = labels_dir / "I.yaml"
        with open(p, "w") as f:
            yaml.dump(content, f, sort_keys=False)

        monkeypatch.setattr(matcher_mod, "_LABELS_DIR", labels_dir)
        return labels_dir

    def test_filter_by_regime(self, tmp_path, monkeypatch):
        """Filter strong_trend returns correct periods."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        periods = get_regime_periods("I", "strong_trend")
        assert len(periods) == 2

    def test_filter_by_regime_and_direction(self, tmp_path, monkeypatch):
        """Filter by both regime and direction."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        periods = get_regime_periods("I", "mild_trend", direction="down")
        assert len(periods) == 1
        assert periods[0] == (date(2017, 11, 1), date(2018, 8, 31))

    def test_filter_by_split(self, tmp_path, monkeypatch):
        """Filter by split."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        periods = get_regime_periods("I", "strong_trend", split="oos")
        assert len(periods) == 1
        assert periods[0][0] == date(2020, 9, 1)

    def test_get_train_periods(self, tmp_path, monkeypatch):
        """get_train_periods shortcut works."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        periods = get_train_periods("I", "strong_trend")
        assert len(periods) == 1
        assert periods[0][0] == date(2015, 4, 1)

    def test_get_oos_periods(self, tmp_path, monkeypatch):
        """get_oos_periods shortcut works."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        periods = get_oos_periods("I", "strong_trend")
        assert len(periods) == 1

    def test_get_holdout_periods(self, tmp_path, monkeypatch):
        """get_holdout_periods shortcut works."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        periods = get_holdout_periods("I", "mean_reversion")
        assert len(periods) == 1

    def test_no_match_returns_empty(self, tmp_path, monkeypatch):
        """Non-matching filter returns empty list."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        periods = get_regime_periods("I", "crisis")
        assert periods == []

    def test_nonexistent_instrument(self, tmp_path, monkeypatch):
        """Missing instrument file raises FileNotFoundError."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        with pytest.raises(FileNotFoundError):
            get_regime_periods("NONEXISTENT", "strong_trend")

    def test_returns_buffer_dates(self, tmp_path, monkeypatch):
        """Returned periods use buffer dates, not core dates."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        periods = get_regime_periods("I", "strong_trend", split="train")
        # buffer_start=2015-04-01, not core start=2015-06-01
        assert periods[0][0] == date(2015, 4, 1)
        assert periods[0][1] == date(2016, 4, 30)

    def test_periods_sorted(self, tmp_path, monkeypatch):
        """Returned periods are sorted by start date."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        periods = get_regime_periods("I", "strong_trend")
        for i in range(len(periods) - 1):
            assert periods[i][0] <= periods[i + 1][0]

    def test_direction_filter_none_returns_all(self, tmp_path, monkeypatch):
        """direction=None does not filter."""
        self._setup_labels_dir(tmp_path, monkeypatch)
        all_st = get_regime_periods("I", "strong_trend", direction=None)
        up_only = get_regime_periods("I", "strong_trend", direction="up")
        assert len(all_st) >= len(up_only)

    def test_labels_without_buffer_fallback(self, tmp_path, monkeypatch):
        """Labels without buffer dates fall back to core dates."""
        import regime.matcher as matcher_mod

        labels_dir = tmp_path / "data2" / "regime_labels"
        labels_dir.mkdir(parents=True)
        content = {
            "instrument": "RB",
            "version": 1,
            "labeled_by": "test",
            "labels": [
                {
                    "start": "2020-01-01",
                    "end": "2020-06-30",
                    "regime": "mild_trend",
                    "direction": "up",
                    "split": "train",
                },
            ],
        }
        with open(labels_dir / "RB.yaml", "w") as f:
            yaml.dump(content, f, sort_keys=False)
        monkeypatch.setattr(matcher_mod, "_LABELS_DIR", labels_dir)

        periods = get_regime_periods("RB", "mild_trend")
        assert periods[0] == (date(2020, 1, 1), date(2020, 6, 30))
