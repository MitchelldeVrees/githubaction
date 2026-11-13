"""Tests for RSI and RSI divergence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest  # noqa: E402
import scanner  # noqa: E402
from test_backtest import random_daily  # noqa: E402
from test_context import trending_daily  # noqa: E402


def candles(closes: list[float]) -> pd.DataFrame:
    close = np.array(closes, dtype=float)
    index = pd.date_range("2020-01-03", periods=len(close), freq="W-FRI")
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1}, index=index
    )


def path(*legs: tuple[float, int]) -> list[float]:
    """Join straight legs, each given as (end price, number of candles)."""
    closes = [100.0]
    for end, count in legs:
        closes.extend(np.linspace(closes[-1], end, count + 1)[1:])
    return closes


def choppy(start: float, end: float, count: int) -> list[float]:
    """A slow rise in which every third candle gives back most of the gain."""
    line = np.linspace(start, end, count + 1)[1:]
    return [value - (3.0 if step % 3 == 2 else 0.0) for step, value in enumerate(line)]


FLAT = (100.0, 30)
FAST_RALLY = (150.0, 12)
PULLBACK = (135.0, 8)


def test_rsi_is_bounded_and_follows_direction():
    rising = scanner.calculate_rsi(pd.Series(np.linspace(100, 200, 60)))
    falling = scanner.calculate_rsi(pd.Series(np.linspace(200, 100, 60)))
    mixed = scanner.calculate_rsi(random_daily(1)["Close"])
    assert rising.iloc[: scanner.RSI_LENGTH].isna().all()
    assert rising.iloc[-1] == 100
    assert falling.iloc[-1] == 0
    assert mixed.dropna().between(0, 100).all()


def test_higher_high_on_weaker_rsi_is_negative_divergence():
    closes = path(FLAT, FAST_RALLY, PULLBACK)
    closes += choppy(closes[-1], 153.0, 24)  # higher high, reached slowly
    closes += list(np.linspace(closes[-1], 140.0, 5)[1:])
    ohlc = candles(closes)

    result = scanner.divergence_context(ohlc, len(ohlc) - 1)
    assert result["negative_divergence"] is True
    assert result["positive_divergence"] is False


def test_higher_high_on_stronger_rsi_is_not_divergence():
    closes = choppy(100.0, 130.0, 30)  # first high, reached slowly
    closes += list(np.linspace(closes[-1], 120.0, 9)[1:])
    closes += list(np.linspace(120.0, 160.0, 11)[1:])  # higher high, reached fast
    closes += list(np.linspace(160.0, 150.0, 5)[1:])
    ohlc = candles([100.0] * 30 + closes)

    assert scanner.divergence_context(ohlc, len(ohlc) - 1)["negative_divergence"] is False


def test_lower_low_on_stronger_rsi_is_positive_divergence():
    mirrored = path(FLAT, FAST_RALLY, PULLBACK)
    mirrored += choppy(mirrored[-1], 153.0, 24)
    mirrored += list(np.linspace(mirrored[-1], 140.0, 5)[1:])
    ohlc = candles([300.0 - value for value in mirrored])

    result = scanner.divergence_context(ohlc, len(ohlc) - 1)
    assert result["positive_divergence"] is True
    assert result["negative_divergence"] is False


def test_stale_swing_is_not_divergence(monkeypatch):
    closes = path(FLAT, FAST_RALLY, PULLBACK)
    closes += choppy(closes[-1], 153.0, 24)
    closes += list(np.linspace(closes[-1], 140.0, 5)[1:])
    ohlc = candles(closes)
    assert scanner.divergence_context(ohlc, len(ohlc) - 1)["negative_divergence"] is True

    monkeypatch.setattr(scanner, "DIVERGENCE_MAX_AGE", 2)
    assert scanner.divergence_context(ohlc, len(ohlc) - 1)["negative_divergence"] is False


def test_divergence_is_unknown_before_rsi_exists():
    ohlc = candles(list(np.linspace(100, 110, 10)))
    result = scanner.divergence_context(ohlc, len(ohlc) - 1)
    assert result["rsi"] is None
    assert result["negative_divergence"] is None


def test_divergence_does_not_depend_on_later_candles():
    ohlc = scanner.resample_ohlc(random_daily(7), "W-FRI")
    for position in range(60, len(ohlc) - 60, 97):
        early = scanner.divergence_context(ohlc.iloc[: position + 1], position)
        late = scanner.divergence_context(ohlc, position)
        assert early == late


def test_alert_shows_divergence_for_its_direction():
    result = scanner.scan_ticker("TEST", daily=trending_daily(320, 0.92))
    assert result["dominant_direction"] == "support"
    assert result["negative_divergence"] in (True, False)
    assert result["rsi"] is not None
    section = scanner.build_signal_telegram_section(result)
    assert "Negative divergence: " in section
    assert "Positive divergence" not in section


def test_backtest_records_and_groups_divergence():
    trades, _ = backtest.backtest_ticker("TEST", random_daily(6))
    assert all(trade.negative_divergence in (True, False, None) for trade in trades)
    assert any(trade.rsi_at_entry is not None for trade in trades)

    summary = backtest.grouped_summary(trades)
    counted = sum(
        summary[name]["closed_trades"] + summary[name]["open_trades"]
        for name in summary
        if name.endswith("divergence")
    )
    known = sum(1 for trade in trades if backtest.confirming_divergence(trade) is not None)
    assert counted == known


def test_swings_are_the_ones_compared():
    closes = path(FLAT, FAST_RALLY, PULLBACK)
    closes += choppy(closes[-1], 153.0, 24)
    closes += list(np.linspace(closes[-1], 140.0, 5)[1:])
    ohlc = candles(closes)
    rsi = scanner.calculate_rsi(ohlc["Close"]).to_numpy()
    high = ohlc["High"].to_numpy()
    found, first, second = scanner.divergence_swings(high, rsi, len(ohlc) - 1, "negative")
    assert found is True
    assert high[second] > high[first] and rsi[second] < rsi[first]

    details = backtest.swing_details(ohlc, len(ohlc) - 1, "short")
    assert details["divergence_swing_1_date"] == ohlc.index[first].date().isoformat()
    assert details["divergence_swing_2_price"] == round(float(high[second]), 4)


def test_backtest_trades_carry_their_swings():
    trades, _ = backtest.backtest_ticker(
        "TEST", random_daily(6), setups=("break", "pullback", "retracement")
    )
    with_divergence = [t for t in trades if backtest.confirming_divergence(t)]
    assert with_divergence
    for trade in with_divergence:
        assert trade.divergence_swing_1_date < trade.divergence_swing_2_date <= trade.entry_date
        if trade.side == "short":
            assert trade.divergence_swing_2_price > trade.divergence_swing_1_price
            assert trade.divergence_swing_2_rsi < trade.divergence_swing_1_rsi
        else:
            assert trade.divergence_swing_2_price < trade.divergence_swing_1_price
            assert trade.divergence_swing_2_rsi > trade.divergence_swing_1_rsi
