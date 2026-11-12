"""Shared test setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scanner  # noqa: E402

ORIGINAL_HIERARCHY = [
    (timeframe, length) for timeframe in scanner.TIMEFRAME_RULES for length in (20, 50)
]


@pytest.fixture(autouse=True)
def original_dominance(monkeypatch, request):
    """Run tests with the original dominance rules unless they ask for the JDB ones.

    Most tests check behavior that does not depend on how dominance is defined,
    and their price series are too smooth to touch an average.
    """
    if request.node.get_closest_marker("jdb_dominance"):
        return
    monkeypatch.setattr(scanner, "MIN_TOUCHES", 0)
    monkeypatch.setattr(scanner, "FAULTY_CANDLES_ALLOWED", 0)
    monkeypatch.setattr(scanner, "HIERARCHY", ORIGINAL_HIERARCHY)
