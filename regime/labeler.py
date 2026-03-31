"""Auto-labeling using a Bry-Boschan variant for regime detection.

Identifies local peaks and troughs in a price series, classifies
the moves between them into regime types, and returns a list of
RegimeLabel objects with buffer windows.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Any

import numpy as np

from config import get_regime_thresholds
from regime.schema import RegimeLabel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_peaks(prices: np.ndarray, window: int) -> np.ndarray:
    """Find indices of local maxima within *window* bars on each side.

    Args:
        prices: 1-D price array.
        window: Number of bars to look left and right.

    Returns:
        Sorted array of peak indices.
    """
    n = len(prices)
    peaks: list[int] = []
    for i in range(window, n - window):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        if prices[i] == np.max(prices[lo:hi]):
            peaks.append(i)
    return np.array(peaks, dtype=int)


def _find_troughs(prices: np.ndarray, window: int) -> np.ndarray:
    """Find indices of local minima within *window* bars on each side.

    Args:
        prices: 1-D price array.
        window: Number of bars to look left and right.

    Returns:
        Sorted array of trough indices.
    """
    n = len(prices)
    troughs: list[int] = []
    for i in range(window, n - window):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        if prices[i] == np.min(prices[lo:hi]):
            troughs.append(i)
    return np.array(troughs, dtype=int)


def _merge_extrema(
    peaks: np.ndarray, troughs: np.ndarray
) -> list[tuple[int, str]]:
    """Merge peaks and troughs into an alternating sequence of (index, type).

    When consecutive extrema are the same type, keep the most extreme one.

    Args:
        peaks: Sorted peak indices.
        troughs: Sorted trough indices.

    Returns:
        List of (bar_index, 'peak' | 'trough') in chronological order.
    """
    combined: list[tuple[int, str]] = []
    combined.extend((int(i), "peak") for i in peaks)
    combined.extend((int(i), "trough") for i in troughs)
    combined.sort(key=lambda x: x[0])
    return combined


def _enforce_alternation(
    extrema: list[tuple[int, str]], prices: np.ndarray
) -> list[tuple[int, str]]:
    """Enforce strict peak-trough alternation.

    When two consecutive peaks occur, keep the higher one.
    When two consecutive troughs occur, keep the lower one.

    Args:
        extrema: Merged list of (index, type).
        prices: Price array for tie-breaking.

    Returns:
        Filtered list with strict alternation.
    """
    if len(extrema) <= 1:
        return list(extrema)

    result: list[tuple[int, str]] = [extrema[0]]

    for idx, typ in extrema[1:]:
        if typ == result[-1][1]:
            # Same type: keep more extreme
            prev_idx, prev_typ = result[-1]
            if typ == "peak":
                if prices[idx] >= prices[prev_idx]:
                    result[-1] = (idx, typ)
            else:
                if prices[idx] <= prices[prev_idx]:
                    result[-1] = (idx, typ)
        else:
            result.append((idx, typ))

    return result


def _compute_rolling_atr(
    prices: np.ndarray, window: int = 20
) -> np.ndarray:
    """Compute a simple rolling ATR proxy using absolute daily returns.

    Args:
        prices: 1-D price array (close prices).
        window: Rolling window size.

    Returns:
        Array of same length as prices with rolling ATR values.
        First *window* values are NaN.
    """
    returns = np.abs(np.diff(prices, prepend=prices[0]))
    atr = np.full_like(prices, np.nan, dtype=float)
    for i in range(window, len(prices)):
        atr[i] = np.mean(returns[i - window + 1 : i + 1])
    return atr


def _add_months(d: date, months: int) -> date:
    """Add (or subtract) *months* from a date, clamping day if needed.

    Args:
        d: The base date.
        months: Number of months to add (can be negative).

    Returns:
        The resulting date.
    """
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, max_day))


def _bars_to_months(n_bars: int, total_bars: int, total_days: int) -> float:
    """Estimate duration in months given number of bars.

    Args:
        n_bars: Number of bars in the segment.
        total_bars: Total bars in the dataset.
        total_days: Total calendar days spanned by the dataset.

    Returns:
        Approximate duration in months.
    """
    if total_bars <= 1:
        return 0.0
    days_per_bar = total_days / max(total_bars - 1, 1)
    return (n_bars * days_per_bar) / 30.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_label(
    prices: np.ndarray,
    dates: np.ndarray,
    config: dict[str, Any] | None = None,
    instrument: str | None = None,
) -> list[RegimeLabel]:
    """Auto-label a price series into regime periods using a Bry-Boschan variant.

    Steps:
        1. Find local peaks / troughs within ``peak_trough_window`` months.
        2. Compute price change between adjacent extrema.
        3. Classify:
           - |change| > strong_trend_pct  ->  strong_trend
           - mild_trend_pct .. strong_trend_pct  ->  mild_trend
           - < mild_trend_pct & duration > 1 month  ->  mean_reversion
        4. Overlay crisis detection: rolling ATR > crisis_atr_sigma * std(ATR).
        5. Enforce minimum duration (min_duration_months).
        6. Add buffer_months before and after each period.

    Args:
        prices: 1-D numpy array of close prices.
        dates: 1-D numpy array of dates (datetime64 or date objects).
                Must be the same length as *prices*.
        config: Optional dict of threshold overrides. When None, loads from
                config/regime_thresholds.yaml.
        instrument: Instrument code for per-instrument threshold overrides.

    Returns:
        List of RegimeLabel instances sorted by start date.
    """
    if len(prices) == 0 or len(dates) == 0:
        return []

    prices = np.asarray(prices, dtype=float)
    n = len(prices)

    # Load thresholds
    if config is None:
        config = get_regime_thresholds(instrument)

    strong_pct: float = config.get("strong_trend_pct", 0.20)
    mild_pct: float = config.get("mild_trend_pct", 0.05)
    crisis_sigma: float = config.get("crisis_atr_sigma", 3.0)
    min_dur: int = config.get("min_duration_months", 1)
    buffer_m: int = config.get("buffer_months", 2)
    pt_window: int = config.get("peak_trough_window", 3)

    # Convert dates to Python date objects
    parsed_dates: list[date] = []
    for d in dates:
        if isinstance(d, date):
            parsed_dates.append(d)
        elif isinstance(d, np.datetime64):
            ts = (d.astype("datetime64[D]") - np.datetime64("1970-01-01", "D")).astype(int)
            parsed_dates.append(date(1970, 1, 1) + timedelta(days=int(ts)))
        else:
            parsed_dates.append(date.fromisoformat(str(d)))

    total_days = max((parsed_dates[-1] - parsed_dates[0]).days, 1)

    # Approximate window in bars (assuming ~21 trading days per month)
    bars_per_month = max(n / max(total_days / 30.0, 1), 1)
    window_bars = max(int(pt_window * bars_per_month), 1)

    # Step 1: Find extrema
    peaks = _find_peaks(prices, window_bars)
    troughs = _find_troughs(prices, window_bars)

    if len(peaks) == 0 and len(troughs) == 0:
        return []

    extrema = _merge_extrema(peaks, troughs)
    extrema = _enforce_alternation(extrema, prices)

    if len(extrema) < 2:
        return []

    # Step 4 (precompute): Crisis detection via rolling ATR
    atr = _compute_rolling_atr(prices, window=max(window_bars, 5))
    valid_atr = atr[~np.isnan(atr)]
    if len(valid_atr) > 0:
        atr_mean = np.mean(valid_atr)
        atr_std = np.std(valid_atr)
        crisis_threshold = atr_mean + crisis_sigma * atr_std
    else:
        crisis_threshold = float("inf")

    # Steps 2-3: Classify segments
    raw_labels: list[RegimeLabel] = []

    for seg_idx in range(len(extrema) - 1):
        idx_a, type_a = extrema[seg_idx]
        idx_b, type_b = extrema[seg_idx + 1]

        seg_start = parsed_dates[idx_a]
        seg_end = parsed_dates[idx_b]

        price_a = prices[idx_a]
        price_b = prices[idx_b]
        if price_a == 0:
            continue

        change = (price_b - price_a) / price_a
        abs_change = abs(change)

        # Duration check
        duration_months = _bars_to_months(idx_b - idx_a, n, total_days)

        # Crisis check: is average ATR in this segment above threshold?
        seg_atr = atr[idx_a : idx_b + 1]
        seg_atr_valid = seg_atr[~np.isnan(seg_atr)]
        is_crisis = (
            len(seg_atr_valid) > 0 and np.mean(seg_atr_valid) > crisis_threshold
        )

        # Classify
        if is_crisis:
            regime = "crisis"
        elif abs_change > strong_pct:
            regime = "strong_trend"
        elif abs_change >= mild_pct:
            regime = "mild_trend"
        elif duration_months >= min_dur:
            regime = "mean_reversion"
        else:
            # Too short and too small — skip
            continue

        # Direction
        if abs_change < mild_pct and regime != "crisis":
            direction = "neutral"
        elif change > 0:
            direction = "up"
        else:
            direction = "down"

        # Duration filter
        if regime != "crisis" and duration_months < min_dur:
            continue

        # Buffer dates
        buf_start = _add_months(seg_start, -buffer_m)
        buf_end = _add_months(seg_end, buffer_m)

        raw_labels.append(
            RegimeLabel(
                start=seg_start,
                end=seg_end,
                regime=regime,
                direction=direction,
                driver="",
                buffer_start=buf_start,
                buffer_end=buf_end,
                split="train",
            )
        )

    return raw_labels
