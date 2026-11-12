"""Tests for the context fields added next to each dominant level."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scanner  # noqa: E402


def last_completed_friday() -> pd.Timestamp:
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    return today - pd.Timedelta(days=(today.weekday() - 4) % 7 or 7)


def trending_daily(weeks: int, final_week_change: float) -> pd.DataFrame:
    """Steady uptrend of `weeks` weeks whose final week moves by the given factor."""
    end = last_completed_friday()
    days = pd.bdate_range(end=end, periods=weeks * 5)
    close = 100 * np.exp(0.005 / 5 * np.arange(len(days)))
    close[-5:] = close[-6] * final_week_change
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=days,
    )


def test_break_reports_next_average_and_ma200():
    result = scanner.scan_ticker("TEST", daily=trending_daily(320, 0.92))

    assert result["signal_type"] == "break_below_dominant_support"
    assert (result["dominant_timeframe"], result["dominant_length"]) == ("Weekly", 20)
    assert result["candle_date"] == last_completed_friday().date().isoformat()

    average_type = result["dominant_average_type"]
    assert result["next_target"] == f"Weekly 50{average_type}"
    assert result["next_target_value"] < result["latest_close"]
    assert result["next_target_distance_pct"] < 0
    assert result["ma200_value"] < result["next_target_value"]
    assert result["ma200_distance_pct"] < result["next_target_distance_pct"]

    section = scanner.build_signal_telegram_section(result)
    assert f"Candle end: {result['candle_date']}" in section
    assert f"Next MA: Weekly 50{average_type}" in section
    assert "MA200: " in section


def test_break_skips_averages_already_passed():
    # A 40% drop closes below the 50 and 200 averages as well.
    result = scanner.scan_ticker("TEST", daily=trending_daily(320, 0.30))

    assert result["signal_type"] == "break_below_dominant_support"
    assert result["next_target"] is None
    assert "Next MA: none in the break direction" in (
        scanner.build_signal_telegram_section(result)
    )


def test_ma200_needs_200_candles():
    result = scanner.scan_ticker("TEST", daily=trending_daily(150, 0.92))

    assert result["signal_type"] == "break_below_dominant_support"
    assert result["ma200_value"] is None
    assert "MA200: n/a (fewer than 200 candles)" in (
        scanner.build_signal_telegram_section(result)
    )


def test_no_signal_row_keeps_context_but_no_target():
    result = scanner.scan_ticker("TEST", daily=trending_daily(320, 1.005))

    assert result["status"] == "no_signal"
    assert result["candle_date"] is not None
    assert result["ma200_value"] is not None
    assert result["next_target"] is None


def test_every_result_row_has_all_fields():
    rows = [
        scanner.scan_ticker("TEST", daily=trending_daily(320, 0.92)),
        scanner.empty_result("TEST", status="error", message="x", error="boom"),
    ]
    for row in rows:
        assert list(row.keys()) == scanner.RESULT_FIELDS
