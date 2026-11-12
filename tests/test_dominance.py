"""Tests for the JDB dominance rules: touches, one faulty candle, MA200."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest
import scanner
from test_backtest import random_daily, scanner_decision

jdb = pytest.mark.jdb_dominance


def weekly_to_daily(close: np.ndarray, low: np.ndarray | None = None) -> pd.DataFrame:
    """Five identical trading days per weekly close, ending on a past Friday."""
    days = pd.bdate_range(end="2026-11-06", periods=len(close) * 5)
    daily_close = np.repeat(close, 5)
    daily_low = daily_close if low is None else np.repeat(low, 5)
    return pd.DataFrame(
        {
            "Open": daily_close,
            "High": daily_close,
            "Low": daily_low,
            "Close": daily_close,
            "Volume": 1000,
        },
        index=days,
    )


def uptrend(weeks: int, touch_every: int | None = 25) -> tuple[np.ndarray, np.ndarray]:
    """Steady weekly uptrend; some weeks dip to the 20-week average intraweek."""
    close = 100 * np.exp(0.005 * np.arange(weeks))
    low = close.copy()
    if touch_every:
        low[::touch_every] *= 0.95
    return close, low


def test_one_close_call_joins_two_runs():
    on_side = np.array([True] * 50 + [False] + [True] * 30)
    close_call = ~on_side
    streak, span, faults = scanner.tolerant_streaks(on_side, close_call)
    assert (streak[49], span[49], faults[49]) == (50, 50, 0)
    assert streak[50] == 0  # the wrong-side candle itself is never in a streak
    assert (streak[-1], span[-1], faults[-1]) == (80, 81, 1)


def test_second_close_call_drops_the_oldest_run():
    on_side = np.array([True] * 50 + [False] + [True] * 30 + [False] + [True] * 10)
    streak, _, faults = scanner.tolerant_streaks(on_side, ~on_side)
    assert (streak[-1], faults[-1]) == (40, 1)


def test_large_miss_or_two_misses_end_the_streak():
    on_side = np.array([True] * 50 + [False] + [True] * 30)
    streak, _, faults = scanner.tolerant_streaks(on_side, np.zeros(len(on_side), dtype=bool))
    assert (streak[-1], faults[-1]) == (30, 0)

    on_side = np.array([True] * 50 + [False, False] + [True] * 30)
    streak, _, _ = scanner.tolerant_streaks(on_side, ~on_side)
    assert streak[-1] == 30


def test_touches_are_counted_inside_the_streak_only():
    touches = pd.Series([1, 0, 1, 0, 0, 1, 0])
    span = np.array([1, 2, 3, 1, 2, 3, 7])
    assert list(scanner.touches_within(touches, span)) == [1, 1, 2, 0, 0, 1, 3]


@jdb
def test_streak_without_touches_is_not_dominant():
    close, low = uptrend(320, touch_every=None)
    close[-1] *= 0.92
    result = scanner.scan_ticker("TEST", daily=weekly_to_daily(close, close))
    assert result["status"] == "no_signal"
    assert result["dominant_timeframe"] is None


@jdb
def test_streak_with_touches_is_dominant_and_breaks():
    close, low = uptrend(320)
    close[-1] *= 0.92
    low[-1] = close[-1]
    result = scanner.scan_ticker("TEST", daily=weekly_to_daily(close, low))
    assert result["signal_type"] == "break_below_dominant_support"
    assert (result["dominant_timeframe"], result["dominant_length"]) == ("Weekly", 20)
    assert result["streak_touches"] >= scanner.MIN_TOUCHES
    assert result["faulty_candles"] == 0
    assert "Touches in streak: " in scanner.build_signal_telegram_section(result)


def trend_with_miss(miss_factor: float) -> pd.DataFrame:
    """Uptrend with one weekly close below the 20-week average, 69 weeks before a break."""
    close, low = uptrend(320)
    close[250] *= miss_factor
    low[250] = close[250]
    close[-1] *= 0.92
    low[-1] = close[-1]
    return weekly_to_daily(close, low)


@jdb
def test_close_call_does_not_reset_dominance():
    result = scanner.scan_ticker("TEST", daily=trend_with_miss(0.945))
    assert result["signal_type"] == "break_below_dominant_support"
    assert result["faulty_candles"] == 1
    assert result["bars_streak"] > 200
    assert "one close call ignored" in scanner.build_signal_telegram_section(result)


@jdb
def test_large_miss_resets_dominance():
    result = scanner.scan_ticker("TEST", daily=trend_with_miss(0.88))
    assert result["signal_type"] is None


@jdb
def test_close_call_resets_dominance_when_not_allowed(monkeypatch):
    monkeypatch.setattr(scanner, "FAULTY_CANDLES_ALLOWED", 0)
    result = scanner.scan_ticker("TEST", daily=trend_with_miss(0.945))
    assert result["signal_type"] is None


@jdb
def test_ma200_can_be_the_dominant_average():
    weeks = np.arange(700)
    close = 100 * np.exp(0.003 * weeks) * (1 + 0.24 * np.sin(2 * np.pi * weeks / 80))
    close[-1] = close[-2] * 0.45
    result = scanner.scan_ticker("TEST", daily=weekly_to_daily(close))
    assert result["signal_type"] == "break_below_dominant_support"
    assert (result["dominant_timeframe"], result["dominant_length"]) == ("Weekly", 200)
    assert result["streak_touches"] >= scanner.MIN_TOUCHES


@jdb
def test_hierarchy_covers_every_dominant_length():
    assert {length for _, length in scanner.HIERARCHY} == set(scanner.DOMINANT_LENGTHS)
    assert scanner.DOMINANT_LENGTHS == (20, 50, 200)


@jdb
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_backtest_signals_match_the_scanner_with_jdb_rules(seed):
    daily = random_daily(seed)
    analyses = scanner.build_candidate_analyses(daily)
    fast = {
        signal.run_date: (
            signal.timeframe,
            signal.length,
            signal.average_type,
            signal.signal_type,
        )
        for signal in backtest.iter_run_signals("TEST", analyses)
    }
    assert fast, "the test data should produce at least one break"

    run_dates = analyses[("Weekly", 20, "SMA")].ohlc.index
    rng = np.random.default_rng(seed)
    quiet = [date for date in run_dates[120:] if date not in fast]
    sample = list(fast)[:25] + list(rng.choice(quiet, size=25, replace=False))
    for run_date in sample:
        assert scanner_decision(daily, pd.Timestamp(run_date)) == fast.get(run_date), run_date
