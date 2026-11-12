"""Tests for the retracement short (JDB component 2) and the data check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest
import scanner
from test_backtest import random_daily
from test_dominance import uptrend, weekly_to_daily

jdb = pytest.mark.jdb_dominance
FIB_STOP = 0.618 + 0.2 * (1 - 0.618)


def path_to(closes: list[float], end: float, weeks: int) -> list[float]:
    return closes + list(np.linspace(closes[-1], end, weeks + 1)[1:])


def scenario(*legs: tuple[float, int]) -> tuple[pd.DataFrame, float]:
    """Uptrend with touches of its 20-week average, a break, then the given legs."""
    close, low = uptrend(320)
    top = float(close.max())
    closes = list(close) + [close[-1] * 0.92]
    for end, weeks in legs:
        closes = path_to(closes, end, weeks)
    closes = np.array(closes)
    lows = np.concatenate([low, closes[len(low) :]])
    return weekly_to_daily(closes, lows), top


def retracements(daily: pd.DataFrame, **settings) -> list[backtest.Trade]:
    trades, _ = backtest.backtest_ticker("TEST", daily, setups=("retracement",), **settings)
    return trades


@jdb
def test_short_at_50_percent_reaches_the_ma200():
    daily, top = scenario((380.0, 10), (450.0, 6), (150.0, 40))
    trades = retracements(daily)
    assert len(trades) == 1
    trade = trades[0]
    assert (trade.setup, trade.side, trade.timeframe) == ("retracement", "short", "Weekly")

    span = top - 380.0
    assert trade.entry_price >= 380.0 + 0.5 * span - 1e-6
    assert trade.entry_price < trade.stop_at_entry
    assert trade.stop_at_entry == pytest.approx(380.0 + FIB_STOP * span, rel=1e-6)
    assert trade.exit_reason == "target"
    assert trade.exit_price == pytest.approx(trade.target_at_entry, rel=0.05)
    assert trade.return_pct > 0 and trade.r_multiple > 1


@jdb
def test_no_immediate_reentry_after_a_stop_without_a_new_low():
    # Stopped on the way up, then price drifts down without a new low and
    # rallies again: no second attempt.
    daily, _ = scenario((380.0, 10), (470.0, 8), (400.0, 6), (445.0, 4), (150.0, 40))
    trades = retracements(daily)
    assert [trade.exit_reason for trade in trades] == ["stop"]


@jdb
def test_rally_through_the_stop_then_a_second_attempt():
    daily, top = scenario((380.0, 10), (470.0, 8), (360.0, 8), (440.0, 6), (150.0, 40))
    trades = retracements(daily)
    assert [trade.exit_reason for trade in trades] == ["stop", "target"]
    first, second = trades
    assert first.exit_price >= first.stop_at_entry  # stop, or the open if it gapped over
    assert first.r_multiple < -0.99
    assert second.entry_date > first.exit_date
    # The second attempt is drawn from the new, lower low.
    assert second.stop_at_entry == pytest.approx(360.0 + FIB_STOP * (top - 360.0), rel=1e-6)


@jdb
def test_no_short_when_price_reaches_the_ma200_first():
    daily, _ = scenario((150.0, 30), (400.0, 10))
    assert retracements(daily) == []


@jdb
def test_no_short_when_price_never_rallies():
    daily, _ = scenario((300.0, 40))
    assert retracements(daily) == []


@jdb
def test_new_high_ends_the_idea():
    daily, top = scenario((400.0, 6), (520.0, 10), (450.0, 3))
    trades = retracements(daily)
    # The rally reaches 50%, is stopped, then makes a new high: no further attempt.
    assert len(trades) <= 1
    assert all(trade.exit_reason == "stop" for trade in trades)


def test_retracements_do_not_depend_on_later_data():
    daily = random_daily(11)
    full = retracements(daily)
    closed = [trade for trade in full if trade.exit_reason != "open"]
    assert closed, "the random series should produce retracement trades"
    for trade in closed[:8]:
        cutoff = pd.Timestamp(trade.exit_date) + pd.Timedelta(days=7)
        assert trade in retracements(daily.loc[:cutoff])


def test_clean_history_drops_data_before_a_bad_jump():
    daily = random_daily(1, years=5)
    broken = daily.copy()
    jump_day = broken.index[300]
    broken.loc[: broken.index[299], ["Open", "High", "Low", "Close"]] /= 16
    cleaned, flags = backtest.clean_history(broken, max_jump=4.0)
    assert [flag["date"] for flag in flags] == [jump_day.date().isoformat()]
    assert flags[0]["ratio"] == pytest.approx(16 * daily["Close"].iloc[300] / daily["Close"].iloc[299], rel=1e-3)
    assert cleaned.index[0] > jump_day
    pd.testing.assert_frame_equal(cleaned, daily.loc[daily.index > jump_day])


def test_clean_history_keeps_normal_data():
    daily = random_daily(2)
    cleaned, flags = backtest.clean_history(daily, max_jump=4.0)
    assert flags == []
    assert cleaned is daily
