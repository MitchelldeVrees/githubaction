"""Tests for the pullback to the dominant average (JDB component 1, signal B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest
import scanner
from test_backtest import random_daily
from test_dominance import uptrend, weekly_to_daily


def trimmed_analyses(daily: pd.DataFrame, run_date: pd.Timestamp):
    """The scanner's analyses when it only knows candles completed by run_date."""
    from dataclasses import fields, replace

    result = {}
    for key, analysis in scanner.build_candidate_analyses(daily.loc[:run_date]).items():
        keep = int((analysis.ohlc.index <= run_date).sum())
        if keep < 2:
            continue
        series = {
            field.name: getattr(analysis, field.name).iloc[:keep]
            for field in fields(analysis)
            if isinstance(getattr(analysis, field.name), (pd.Series, pd.DataFrame))
        }
        result[key] = replace(analysis, **series)
    return result


@pytest.mark.parametrize("seed", [1, 2])
def test_dominant_levels_match_the_scanner(seed):
    daily = random_daily(seed)
    analyses = scanner.build_candidate_analyses(daily)
    fast = {
        level.run_date: (level.timeframe, level.length, level.average_type, level.direction)
        for level in backtest.iter_dominant_levels("TEST", analyses, every_run_date=True)
    }
    assert fast

    run_dates = analyses[("Weekly", 20, "SMA")].ohlc.index
    rng = np.random.default_rng(seed)
    for run_date in rng.choice(run_dates[150:], size=40, replace=False):
        run_date = pd.Timestamp(run_date)
        dominant = scanner.select_dominant_level(
            "TEST", trimmed_analyses(daily, run_date), as_of_pos=-1
        )
        expected = None
        if dominant is not None:
            expected = (
                dominant.timeframe,
                dominant.length,
                dominant.average_type,
                1 if dominant.direction == "support" else -1,
            )
        assert fast.get(run_date) == expected, run_date


def weekly_trades(close, low, **settings):
    analyses = scanner.build_candidate_analyses(weekly_to_daily(close, low))
    return backtest.find_pullbacks("TEST", analyses, **settings)


def test_touch_of_dominant_support_is_a_long_that_trails():
    close, low = uptrend(320)  # every 25th week dips to just below its 20-week average
    close[-1] *= 0.80
    low[-1] = close[-1]
    trades = weekly_trades(close, low, max_candles=1000)

    first = trades[0]
    assert (first.setup, first.side, first.timeframe, first.length) == (
        "pullback",
        "long",
        "Weekly",
        20,
    )
    assert first.target == "none"  # the 50 and 200 averages are below the price
    assert first.entry_price > first.stop_at_entry
    assert first.exit_reason == "trail"
    assert first.return_pct > 0
    assert first.r_multiple > 1


def test_deep_dip_enters_and_stops_in_the_same_candle():
    close, low = uptrend(320, touch_every=None)
    low[200] = close[200] * 0.90  # far through the average and the stop
    first = weekly_trades(close, low)[0]

    assert first.entry_date == weekly_to_daily(close).index[200 * 5 + 4].date().isoformat()
    assert (first.exit_reason, first.candles_held) == ("stop", 0)
    assert first.exit_price == first.stop_at_entry
    risk = first.entry_price - first.stop_at_entry
    costs_in_r = 2 * 0.1 / 100 * first.entry_price / risk
    assert first.r_multiple == pytest.approx(-1 - costs_in_r, abs=0.01)


def test_no_trade_when_price_stays_away_from_the_average():
    close, low = uptrend(320, touch_every=None)
    assert weekly_trades(close, low) == []


def test_no_trade_when_the_candle_opens_beyond_the_stop():
    close, low = uptrend(320, touch_every=None)
    close[200] *= 0.80  # weekly_to_daily opens the week at its close: a gap far below
    low[200] = close[200]
    entries = {trade.entry_date for trade in weekly_trades(close, low)}
    assert weekly_to_daily(close).index[200 * 5 + 4].date().isoformat() not in entries


def test_attempts_pause_after_three_losses_in_a_row():
    close, low = uptrend(320, touch_every=None)
    for week in range(200, 210):
        low[week] = close[week] * 0.90
    trades = weekly_trades(close, low)
    assert len(trades) == 3
    assert all(trade.return_pct < 0 for trade in trades)

    more = weekly_trades(close, low, max_attempts=5)
    assert len(more) == 5


def test_pullbacks_do_not_depend_on_later_data():
    daily = random_daily(5)
    settings = {"setups": ("pullback",)}
    full, _ = backtest.backtest_ticker("TEST", daily, **settings)
    closed = [trade for trade in full if trade.exit_reason != "open"]
    assert len(closed) > 20
    for trade in closed[::7][:12]:
        cutoff = pd.Timestamp(trade.exit_date) + pd.Timedelta(days=7)
        partial, _ = backtest.backtest_ticker("TEST", daily.loc[:cutoff], **settings)
        assert trade in partial


def test_one_position_per_level_at_a_time():
    daily = random_daily(6)
    trades, _ = backtest.backtest_ticker("TEST", daily, setups=("pullback",))
    by_level: dict[tuple, list] = {}
    for trade in trades:
        by_level.setdefault((trade.timeframe, trade.length, trade.average_type), []).append(trade)
    for level_trades in by_level.values():
        for earlier, later in zip(level_trades, level_trades[1:]):
            assert later.entry_date > earlier.exit_date or earlier.candles_held == 0


def test_summary_separates_breaks_and_pullbacks():
    trades, _ = backtest.backtest_ticker("TEST", random_daily(6), setups=("break", "pullback"))
    summary = backtest.grouped_summary(trades)
    assert {"break short", "pullback long"} <= set(summary)
    total = sum(
        summary[name]["closed_trades"] + summary[name]["open_trades"]
        for name in ("break long", "break short", "pullback long", "pullback short")
        if name in summary
    )
    assert total == len(trades)


def test_entry_and_stop_are_set_in_atr_of_the_signal_timeframe():
    close, low = uptrend(320)
    analyses = scanner.build_candidate_analyses(weekly_to_daily(close, low))
    trade = backtest.find_pullbacks("TEST", analyses, entry_atr=0.25, stop_atr=0.75)[0]

    analysis = analyses[(trade.timeframe, trade.length, trade.average_type)]
    known = analysis.ohlc.index.get_loc(pd.Timestamp(trade.entry_date)) - 1
    average = analysis.ma.iloc[known]
    atr = backtest.calculate_atr(analysis.ohlc)[known]
    assert trade.stop_at_entry == pytest.approx(average - 0.75 * atr, rel=1e-4)
    assert trade.entry_price <= average + 0.25 * atr + 1e-9


def test_volatile_stock_gets_a_wider_stop():
    rng = np.random.default_rng(3)
    close, _ = uptrend(320)
    calm = close * (1 + rng.normal(0, 0.002, len(close)))
    wild = close * (1 + rng.normal(0, 0.02, len(close)))
    distance = {}
    for name, series in (("calm", calm), ("wild", wild)):
        ohlc = scanner.resample_ohlc(weekly_to_daily(series), "W-FRI")
        distance[name] = backtest.calculate_atr(ohlc)[-1] / series[-1]
    assert distance["wild"] > 5 * distance["calm"]


def test_atr_does_not_depend_on_later_candles():
    ohlc = scanner.resample_ohlc(random_daily(7), "W-FRI")
    full = backtest.calculate_atr(ohlc)
    for position in (20, 200, 900):
        assert backtest.calculate_atr(ohlc.iloc[: position + 1])[-1] == pytest.approx(
            full[position]
        )


def test_sma_and_ema_of_one_length_share_the_attempts(monkeypatch):
    """The NVDA 2006 case: three EMA50 losses must not leave room for an SMA50 try."""
    setups = [
        backtest.Setup("TEST", pd.Timestamp("2006-01-06"), "Weekly", 50, kind, 1, position)
        for position, kind in zip(range(100, 106), ["EMA", "EMA", "EMA", "SMA", "SMA", "EMA"])
    ]
    monkeypatch.setattr(backtest, "iter_dominant_levels", lambda *args, **kwargs: iter(setups))

    def losing_trade(setup, *args, **kwargs):
        return backtest.Trade(
            setup="pullback", ticker="TEST", side="long", timeframe=setup.timeframe,
            length=setup.length, average_type=setup.average_type, bars_streak=80,
            streak_touches=5, faulty_candles=0, score=1.0, rsi_at_entry=45.0,
            negative_divergence=False, positive_divergence=False,
            entry_date=str(setup.position), entry_price=1.0, target="none",
            target_at_entry=None, stop_at_entry=0.97, exit_date=str(setup.position),
            exit_price=0.97, exit_reason="stop", candles_held=0, return_pct=-3.2,
            r_multiple=-1.07,
        )

    monkeypatch.setattr(backtest, "simulate_pullback", losing_trade)
    trades = backtest.find_pullbacks("TEST", analyses={})
    assert [trade.average_type for trade in trades] == ["EMA", "EMA", "EMA"]


def test_pullback_without_target_rides_the_trend():
    close, low = uptrend(320)
    analyses = scanner.build_candidate_analyses(weekly_to_daily(close, low))
    with_target = backtest.find_pullbacks("TEST", analyses, max_candles=1000)
    riding = backtest.find_pullbacks("TEST", analyses, max_candles=1000, use_target=False)
    assert riding and all(trade.target == "none" for trade in riding)
    assert {trade.exit_reason for trade in riding} <= {"stop", "trail", "time", "open"}
    assert len(riding) <= len(with_target)


def test_pullback_controls_enter_only_while_the_level_is_dominant():
    daily = random_daily(6)
    analyses = scanner.build_candidate_analyses(daily)
    trades = backtest.find_pullbacks("TEST", analyses, use_target=False)
    rows = backtest.pullback_controls(trades, analyses, 3, 0.75, 104, 0.1, use_target=False)
    assert rows
    for row in rows[:40]:
        analysis = analyses[(row["timeframe"], row["length"], row["average_type"])]
        position = analysis.ohlc.index.get_loc(pd.Timestamp(row["entry_date"]))
        wanted = backtest.SUPPORT if row["side"] == "long" else backtest.RESISTANCE
        assert analysis.direction.iloc[position] == wanted
    assert {row["exit_reason"] for row in rows} <= {"stop", "trail", "time"}
