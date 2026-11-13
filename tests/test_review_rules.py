"""Tests for the stricter rules that came out of reviewing trades on the chart."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest  # noqa: E402
import scanner  # noqa: E402

# --- divergence -------------------------------------------------------------


def swing_series(peaks: dict[int, tuple[float, float]], length: int = 60):
    """Prices and RSI that are low everywhere except at the given peaks."""
    price = np.full(length, 90.0)
    rsi = np.full(length, 40.0)
    for index, (value, strength) in peaks.items():
        price[index] = value
        rsi[index] = strength
    return price, rsi


def test_divergence_needs_the_minimum_gap(monkeypatch):
    price, rsi = swing_series({20: (100.0, 70.0), 40: (103.0, 66.0), 46: (104.0, 64.0)})
    assert scanner.divergence_swings(price, rsi, 55, "negative") == (True, 40, 46)
    monkeypatch.setattr(scanner, "DIVERGENCE_MIN_GAP", 10)
    # The swing at 40 is too close, so 46 is compared with the one at 20.
    assert scanner.divergence_swings(price, rsi, 55, "negative") == (True, 20, 46)
    monkeypatch.setattr(scanner, "DIVERGENCE_MIN_GAP", 30)
    assert scanner.divergence_swings(price, rsi, 55, "negative") == (False, None, None)


def test_divergence_needs_the_minimum_price_and_rsi_difference(monkeypatch):
    price, rsi = swing_series({20: (100.0, 70.0), 40: (100.5, 69.0)})
    assert scanner.divergence_swings(price, rsi, 50, "negative")[0] is True
    monkeypatch.setattr(scanner, "DIVERGENCE_MIN_PRICE_PCT", 1.0)
    assert scanner.divergence_swings(price, rsi, 50, "negative")[0] is False
    monkeypatch.setattr(scanner, "DIVERGENCE_MIN_PRICE_PCT", 0.0)
    monkeypatch.setattr(scanner, "DIVERGENCE_MIN_RSI_DIFF", 2.0)
    assert scanner.divergence_swings(price, rsi, 50, "negative")[0] is False


def test_positive_divergence_uses_the_same_minimums(monkeypatch):
    price = np.full(60, 110.0)
    rsi = np.full(60, 60.0)
    price[20], rsi[20] = 100.0, 30.0
    price[40], rsi[40] = 97.0, 35.0
    monkeypatch.setattr(scanner, "DIVERGENCE_MIN_PRICE_PCT", 2.0)
    monkeypatch.setattr(scanner, "DIVERGENCE_MIN_RSI_DIFF", 4.0)
    assert scanner.divergence_swings(price, rsi, 50, "positive") == (True, 20, 40)
    monkeypatch.setattr(scanner, "DIVERGENCE_MIN_RSI_DIFF", 6.0)
    assert scanner.divergence_swings(price, rsi, 50, "positive")[0] is False


def test_close_source_ignores_wicks(monkeypatch):
    index = pd.date_range("2020-01-03", periods=3, freq="W-FRI")
    ohlc = pd.DataFrame(
        {"Open": 1.0, "High": [5.0, 6.0, 7.0], "Low": [0.5, 0.4, 0.3], "Close": [1.0, 2.0, 3.0]},
        index=index,
    )
    assert list(scanner.divergence_price(ohlc, "negative")) == [5.0, 6.0, 7.0]
    assert list(scanner.divergence_price(ohlc, "positive")) == [0.5, 0.4, 0.3]
    monkeypatch.setattr(scanner, "DIVERGENCE_SOURCE", "close")
    assert list(scanner.divergence_price(ohlc, "negative")) == [1.0, 2.0, 3.0]
    assert list(scanner.divergence_price(ohlc, "positive")) == [1.0, 2.0, 3.0]


# --- dominance ---------------------------------------------------------------


@pytest.mark.jdb_dominance
def test_recent_close_call_is_not_tolerated():
    on_side = np.array([True] * 50 + [False] + [True] * 30)
    streak, span, faults = scanner.tolerant_streaks(on_side, ~on_side, clean_final=10)
    # Within 10 candles of the close call the streak counts from after it.
    assert (streak[60], faults[60]) == (10, 0)
    # Once it is older, the two runs are joined as before.
    assert (streak[61], span[61], faults[61]) == (61, 62, 1)
    assert (streak[-1], faults[-1]) == (80, 1)


def test_recent_touches_can_be_ignored():
    touches = pd.Series([1, 0, 1, 0, 1, 1, 1])
    span = np.full(7, 7)
    assert scanner.touches_within(touches, span)[-1] == 5
    assert scanner.touches_within(touches, span, ignore_last=3)[-1] == 2
    assert scanner.touches_within(touches, span, ignore_last=10)[-1] == 0


def test_touching_candles_in_a_row_count_once():
    touches = pd.Series([False, True, True, True, False, True, False, True, True])
    events = scanner.touch_events(touches)
    assert list(events) == [False, True, False, False, False, True, False, True, False]


@pytest.mark.jdb_dominance
def test_touches_only_at_the_end_of_a_streak_can_be_refused(monkeypatch):
    # A long rise far above the average, then a flat stretch in which the
    # average catches up and every candle touches it.
    close = np.concatenate([np.linspace(100, 200, 120), np.full(25, 200.0)])
    index = pd.date_range("2015-01-02", periods=len(close), freq="W-FRI")
    ohlc = pd.DataFrame(
        {"Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close, "Volume": 1},
        index=index,
    )
    candle = 130
    assert scanner.analyze_candidate(ohlc, "Weekly", 20, "SMA").direction.iloc[candle] == 1

    monkeypatch.setattr(scanner, "TOUCH_CLUSTERS", True)  # one long touch is one touch
    assert scanner.analyze_candidate(ohlc, "Weekly", 20, "SMA").direction.iloc[candle] == 0

    monkeypatch.setattr(scanner, "TOUCH_CLUSTERS", False)
    monkeypatch.setattr(scanner, "TOUCH_IGNORE_LAST", 10)  # the touches are all recent
    assert scanner.analyze_candidate(ohlc, "Weekly", 20, "SMA").direction.iloc[candle] == 0


# --- break trades --------------------------------------------------------------


BREAK = 60  # the break candle; the 50-candle target needs that much history


def break_setup(highs, lows, closes, ma_value=100.0):
    """A short break at candle BREAK of a flat market with a fixed average."""
    count = len(closes)
    index = pd.date_range("2018-01-05", periods=count, freq="W-FRI")
    close = np.asarray(closes, dtype=float)
    ohlc = pd.DataFrame(
        {
            "Open": close,
            "High": np.asarray(highs, dtype=float),
            "Low": np.asarray(lows, dtype=float),
            "Close": close,
            "Volume": 1,
        },
        index=index,
    )
    analysis = SimpleNamespace(ohlc=ohlc, ma=pd.Series(ma_value, index=index))
    signal = backtest.Signal(
        ticker="TEST",
        run_date=index[BREAK],
        timeframe="Weekly",
        length=20,
        average_type="SMA",
        signal_type="break_below_dominant_support",
        position=BREAK,
        bars_streak=80,
        streak_touches=3,
        faulty_candles=0,
        score=1.0,
    )
    return signal, {("Weekly", 20, "SMA"): analysis}


def fake_targets(monkeypatch, value):
    def average(close, length, average_type):
        return pd.Series(value if length == 50 else np.nan, index=close.index)

    monkeypatch.setattr(scanner, "calculate_average", average)


def run_break(signal, analyses, **kwargs):
    return backtest.simulate_trade(signal, analyses, "candle", 104, 0.0, 0.5, **kwargs)


def test_close_stop_survives_a_wick(monkeypatch):
    fake_targets(monkeypatch, 90.0)
    count = 30
    closes = [101.0] * BREAK + [98.0] * count
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    highs[BREAK + 2] = 105.0  # a wick above the stop (99.0 * 1.005), closing back at 98
    lows[BREAK + 5] = 89.0  # then the target
    signal, analyses = break_setup(highs, lows, closes)

    intrabar = run_break(signal, analyses)
    on_close = run_break(signal, analyses, stop_trigger="close")
    stop_date = analyses[("Weekly", 20, "SMA")].ohlc.index[BREAK + 2].date().isoformat()
    assert intrabar.exit_reason == "stop" and intrabar.exit_date == stop_date
    assert on_close.exit_reason == "target" and on_close.exit_price == 90.0


def test_close_stop_exits_at_the_close(monkeypatch):
    fake_targets(monkeypatch, 90.0)
    closes = [101.0] * BREAK + [98.0, 98.0, 103.0] + [98.0] * 10
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    signal, analyses = break_setup(highs, lows, closes)
    trade = run_break(signal, analyses, stop_trigger="close")
    assert trade.exit_reason == "stop" and trade.exit_price == 103.0


def test_break_is_skipped_when_its_candle_hit_the_target(monkeypatch):
    fake_targets(monkeypatch, 90.0)
    closes = [101.0] * BREAK + [98.0] * 10
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    lows[BREAK] = 89.5  # the break candle's wick already reached the target
    signal, analyses = break_setup(highs, lows, closes)
    assert run_break(signal, analyses) is not None
    assert run_break(signal, analyses, skip_target_hit=True) is None


def test_break_is_skipped_when_it_closes_far_from_the_average(monkeypatch):
    fake_targets(monkeypatch, 80.0)
    closes = [101.0] * BREAK + [92.0] * 10  # closes 8 below the average
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    signal, analyses = break_setup(highs, lows, closes)
    assert run_break(signal, analyses, entry_max_atr=20.0) is not None
    assert run_break(signal, analyses, entry_max_atr=1.5) is None


# --- pullbacks -----------------------------------------------------------------


def test_pullbacks_can_be_limited_to_some_lengths(monkeypatch):
    setups = [
        backtest.Setup("TEST", pd.Timestamp("2020-01-03"), "Weekly", length, "EMA", 1, position)
        for position, length in zip(range(100, 103), (20, 50, 200))
    ]
    monkeypatch.setattr(backtest, "iter_dominant_levels", lambda *args, **kwargs: iter(setups))
    seen: list[int] = []
    received: dict = {}

    def record(setup, *args, **kwargs):
        seen.append(setup.length)
        received.update(kwargs)
        return None

    monkeypatch.setattr(backtest, "simulate_pullback", record)
    backtest.find_pullbacks(
        "TEST", {}, lengths=(20, 50), stop_trigger="close", skip_divergence=True
    )
    assert seen == [20, 50]
    assert received == {"stop_trigger": "close", "skip_divergence": True}
