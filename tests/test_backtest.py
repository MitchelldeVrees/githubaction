"""Tests for backtest.py: signals must match the scanner, trades must be causal."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest  # noqa: E402
import scanner  # noqa: E402


def random_daily(seed: int, years: int = 30) -> pd.DataFrame:
    """Trending random walk with occasional sharp moves, ending last Friday."""
    rng = np.random.default_rng(seed)
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    end = today - pd.Timedelta(days=(today.weekday() - 4) % 7 or 7)
    days = pd.bdate_range(end=end, periods=years * 260)
    steps = rng.normal(0.0006, 0.012, len(days))
    shocks = rng.random(len(days)) < 0.002
    steps[shocks] += rng.normal(0, 0.12, shocks.sum())
    close = 50 * np.exp(np.cumsum(steps))
    spread = np.abs(rng.normal(0, 0.008, len(days)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.004, len(days))),
            "High": close * (1 + spread),
            "Low": close * (1 - spread),
            "Close": close,
            "Volume": 1000,
        },
        index=days,
    )


def scanner_decision(daily: pd.DataFrame, run_date: pd.Timestamp):
    """What the scanner itself decides when it only knows data up to run_date."""
    analyses = scanner.build_candidate_analyses(daily.loc[:run_date])
    trimmed = {}
    for key, analysis in analyses.items():
        keep = int((analysis.ohlc.index <= run_date).sum())
        if keep < 2:
            continue
        trimmed[key] = replace(
            analysis,
            ohlc=analysis.ohlc.iloc[:keep],
            ma=analysis.ma.iloc[:keep],
            direction=analysis.direction.iloc[:keep],
            score=analysis.score.iloc[:keep],
            touch_count=analysis.touch_count.iloc[:keep],
            bars_streak=analysis.bars_streak.iloc[:keep],
            streak_touches=analysis.streak_touches.iloc[:keep],
            faulty_candles=analysis.faulty_candles.iloc[:keep],
        )
    dominant = scanner.select_dominant_level("TEST", trimmed, as_of_pos=-2)
    if dominant is None:
        return None
    signal_type = scanner.detect_break_signal(dominant)
    if signal_type is None:
        return None
    return (dominant.timeframe, dominant.length, dominant.average_type, signal_type)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_signals_match_the_scanner(seed):
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


def test_each_break_is_reported_once():
    analyses = scanner.build_candidate_analyses(random_daily(4))
    signals = backtest.find_signals("TEST", analyses)
    keys = [(signal.timeframe, signal.position) for signal in signals]
    assert len(keys) == len(set(keys))


def test_trades_do_not_depend_on_later_data():
    daily = random_daily(5)
    full, _ = backtest.backtest_ticker("TEST", daily)
    closed = [trade for trade in full if trade.exit_reason != "open"]
    assert closed
    for trade in closed[:10]:
        cutoff = pd.Timestamp(trade.exit_date) + pd.Timedelta(days=7)
        partial, _ = backtest.backtest_ticker("TEST", daily.loc[:cutoff])
        assert trade in partial


def steady_trend_then_drop(change: float, weeks_after: int, drift_after: float) -> pd.DataFrame:
    weeks = 320 + weeks_after
    days = pd.bdate_range(end="2026-11-06", periods=weeks * 5)
    close = 100 * np.exp(0.001 * np.arange(len(days)))
    at = 320 * 5
    close[at - 5 : at] = close[at - 6] * change
    close[at:] = close[at - 1] * np.exp(drift_after * np.arange(1, len(days) - at + 1))
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=days,
    )


def test_short_reaches_target():
    trades, skipped = backtest.backtest_ticker(
        "TEST", steady_trend_then_drop(0.92, weeks_after=30, drift_after=-0.004)
    )
    first = trades[0]
    assert (first.side, first.timeframe, first.length) == ("short", "Weekly", 20)
    assert first.target.startswith("Weekly 50")
    assert first.exit_reason == "target"
    assert first.return_pct > 0
    assert first.exit_price < first.entry_price


def test_short_is_stopped_when_price_recovers():
    trades, _ = backtest.backtest_ticker(
        "TEST", steady_trend_then_drop(0.92, weeks_after=30, drift_after=0.004)
    )
    first = trades[0]
    assert first.exit_reason == "stop"
    assert first.return_pct < 0
    assert first.r_multiple == pytest.approx(-1, abs=0.6)


def test_break_without_remaining_average_is_skipped():
    trades, skipped = backtest.backtest_ticker(
        "TEST", steady_trend_then_drop(0.30, weeks_after=4, drift_after=0.0)
    )
    assert skipped >= 1
    assert all(trade.entry_date != "2026-10-09" for trade in trades)


def test_summary_counts():
    trades, _ = backtest.backtest_ticker("TEST", random_daily(6))
    summary = backtest.grouped_summary(trades)
    overall = summary["all"]
    assert overall["closed_trades"] + overall["open_trades"] == len(trades)
    assert sum(overall["exits"].values()) == overall["closed_trades"]


def trade_key(trade):
    return (trade.ticker, trade.timeframe, trade.length, trade.average_type, trade.entry_date)


def test_trailing_stop_only_tightens_the_candle_stop():
    trailed_exits = 0
    for seed in (5, 6, 7, 8):
        daily = random_daily(seed)
        fixed = {trade_key(t): t for t in backtest.backtest_ticker("TEST", daily)[0]}
        trail = backtest.backtest_ticker("TEST", daily, stop_mode="trail")[0]
        assert {trade_key(t) for t in trail} == set(fixed)

        for trade in trail:
            same_signal = fixed[trade_key(trade)]
            assert trade.stop_at_entry == same_signal.stop_at_entry
            if trade.exit_reason != "trail":
                assert trade == same_signal
                continue
            trailed_exits += 1
            assert trade.exit_date <= same_signal.exit_date
            if trade.side == "short":
                assert trade.exit_price <= trade.stop_at_entry
            else:
                assert trade.exit_price >= trade.stop_at_entry
    assert trailed_exits > 0


def test_trailing_stop_does_not_depend_on_later_data():
    daily = random_daily(5)
    full, _ = backtest.backtest_ticker("TEST", daily, stop_mode="trail")
    closed = [trade for trade in full if trade.exit_reason == "trail"]
    assert closed
    for trade in closed[:10]:
        cutoff = pd.Timestamp(trade.exit_date) + pd.Timedelta(days=7)
        partial, _ = backtest.backtest_ticker("TEST", daily.loc[:cutoff], stop_mode="trail")
        assert trade in partial


def test_trailing_waits_for_the_required_profit():
    daily = random_daily(5)
    never = backtest.backtest_ticker("TEST", daily, stop_mode="trail", trail_after_r=1e9)[0]
    fixed = backtest.backtest_ticker("TEST", daily)[0]
    assert never == fixed
