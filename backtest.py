"""Backtest of the dominant MA/EMA break signals raised by scanner.py.

The backtest replays the scanner over history instead of re-implementing it:
it imports the scanner's candle building and dominance analysis, finds every
break the Friday run would have alerted on, and simulates one trade per break.

Trade rules (JDB component 1, break of the dominant average):

- Entry at the close of the candle that breaks the dominant average. A break
  below support is a short, a break above resistance is a long.
- Target is the next longer average (20 -> 50 -> 200) on the same timeframe
  and of the same SMA/EMA type that lies in the direction of the break at
  entry. The target moves with that average; the trade exits when a candle
  touches it. Breaks without such an average are counted but not traded.
- Stop, chosen with --stop:
    candle   beyond the extreme of the break candle, checked inside each candle
    reclaim  a candle closing back across the dominant average
    trail    starts as the candle stop; once a candle has closed at least
             --trail-after-r times the initial risk in profit, the stop follows
             the dominant average (just beyond it) and only ever tightens
- A trade still open after --max-candles candles exits at that candle's close.
  A trade still open at the end of the data is reported as open.

Retracement shorts (JDB component 2) follow a break below dominant support,
see simulate_retracement: short at the 50% retracement of the move from the
recent top to the recent low, stop at about 69.4%, target the 200-period
average.

Before backtesting, each ticker's history is checked for implausible one-day
price jumps (data errors); only the data after the last such jump is used.

Pullback trades (JDB component 1, signal B) are simulated as well, see
simulate_pullback: while an average is dominant, price coming back to it is an
entry just on the trend side of the average with a stop just beyond it.

Each trade also records RSI and RSI divergence on the break candle, as defined
in scanner.find_divergence. Divergence does not filter any trade; the summary
compares trades with and without divergence in the trade's direction.

Everything is decided on completed candles and each exit level is known before
the candle it is tested on. When a candle touches both stop and target, the
stop is assumed to have come first.

This is a research tool, not financial advice. Results are optimistic to the
extent that tickers.txt only holds companies that still exist today.
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import scanner

TARGET_LENGTHS = (20, 50, 200)
ATR_LENGTH = 14
SUPPORT = 1
RESISTANCE = -1


@dataclass(frozen=True)
class Signal:
    """One break the scanner would have alerted on."""

    ticker: str
    run_date: pd.Timestamp
    timeframe: str
    length: int
    average_type: str
    signal_type: str
    position: int  # index of the break candle in the timeframe's candles
    bars_streak: int
    streak_touches: int
    faulty_candles: int
    score: float


@dataclass
class Trade:
    """Outcome of trading one signal."""

    setup: str  # "break" or "pullback"
    ticker: str
    side: str
    timeframe: str
    length: int
    average_type: str
    bars_streak: int
    streak_touches: int
    faulty_candles: int
    score: float
    rsi_at_entry: float | None
    negative_divergence: bool | None
    positive_divergence: bool | None
    entry_date: str
    entry_price: float
    target: str
    target_at_entry: float | None
    stop_at_entry: float | None
    exit_date: str
    exit_price: float
    exit_reason: str
    candles_held: int
    return_pct: float
    r_multiple: float | None


def candidate_arrays(
    analyses: dict[tuple[str, int, str], scanner.CandidateAnalysis],
) -> dict[tuple[str, int, str], dict[str, np.ndarray]]:
    """Convert the scanner's candidate series to arrays for fast lookups."""
    return {
        key: {
            "close": analysis.ohlc["Close"].to_numpy(dtype=float),
            "ma": analysis.ma.to_numpy(dtype=float),
            "direction": analysis.direction.to_numpy(),
            "score": analysis.score.to_numpy(dtype=float),
            "streak": analysis.bars_streak.to_numpy(),
            "touches": analysis.streak_touches.to_numpy(),
            "faults": analysis.faulty_candles.to_numpy(),
        }
        for key, analysis in analyses.items()
    }


def iter_run_signals(
    ticker: str,
    analyses: dict[tuple[str, int, str], scanner.CandidateAnalysis],
) -> Iterator[Signal]:
    """Yield the break signal of every weekly run date, as the scanner decides it.

    For each run date this mirrors scanner.select_dominant_level(as_of_pos=-2)
    followed by scanner.detect_break_signal, using only candles completed on or
    before that date. A monthly or quarterly break is yielded on every run date
    until its next candle completes, exactly like the live scanner; use
    find_signals for one signal per break.
    """
    weekly = next(
        (a for (timeframe, _, _), a in analyses.items() if timeframe == "Weekly"),
        None,
    )
    if weekly is None:
        return

    arrays = candidate_arrays(analyses)
    candle_ends = {
        timeframe: next(
            a.ohlc.index.to_numpy() for (tf, _, _), a in analyses.items() if tf == timeframe
        )
        for timeframe in {key[0] for key in analyses}
    }
    run_dates = weekly.ohlc.index
    positions = {
        timeframe: np.searchsorted(ends, run_dates.to_numpy(), side="right") - 1
        for timeframe, ends in candle_ends.items()
    }

    for run_index, run_date in enumerate(run_dates):
        for timeframe, length in scanner.HIERARCHY:
            if timeframe not in positions:
                continue
            latest = int(positions[timeframe][run_index])
            previous = latest - 1
            if previous < 0:
                continue

            best: tuple[str, dict[str, np.ndarray]] | None = None
            for average_type in scanner.AVERAGE_TYPES:
                data = arrays.get((timeframe, length, average_type))
                if data is None or data["direction"][previous] == 0:
                    continue
                if not (
                    np.isfinite(data["ma"][previous])
                    and np.isfinite(data["ma"][latest])
                    and data["score"][previous] > 0
                ):
                    continue
                if best is None or data["score"][previous] > best[1]["score"][previous]:
                    best = (average_type, data)

            if best is None:
                continue  # no dominant level at this step, try the next one

            average_type, data = best
            direction = int(data["direction"][previous])
            close, ma = data["close"], data["ma"]
            signal_type = None
            if direction == SUPPORT and close[previous] >= ma[previous] and close[latest] < ma[latest]:
                signal_type = "break_below_dominant_support"
            elif (
                direction == RESISTANCE
                and close[previous] <= ma[previous]
                and close[latest] > ma[latest]
            ):
                signal_type = "break_above_dominant_resistance"

            if signal_type is not None:
                yield Signal(
                    ticker=ticker,
                    run_date=run_date,
                    timeframe=timeframe,
                    length=length,
                    average_type=average_type,
                    signal_type=signal_type,
                    position=latest,
                    bars_streak=int(data["streak"][previous]),
                    streak_touches=int(data["touches"][previous]),
                    faulty_candles=int(data["faults"][previous]),
                    score=float(data["score"][previous]),
                )
            break  # the first hierarchy step with a dominant level decides


@dataclass(frozen=True)
class Setup:
    """A dominant level at the moment one of its candles has just completed."""

    ticker: str
    run_date: pd.Timestamp
    timeframe: str
    length: int
    average_type: str
    direction: int  # SUPPORT or RESISTANCE
    position: int  # index of the last completed candle; a pullback enters in the next


def iter_dominant_levels(
    ticker: str,
    analyses: dict[tuple[str, int, str], scanner.CandidateAnalysis],
    every_run_date: bool = False,
) -> Iterator[Setup]:
    """Yield the scanner's dominant level each time a new candle of it completes.

    For each weekly run date this mirrors scanner.select_dominant_level with
    as_of_pos=-1, using only candles completed on or before that date. A level
    is yielded once per candle of its own timeframe, or on every run date with
    `every_run_date`.
    """
    weekly = next(
        (a for (timeframe, _, _), a in analyses.items() if timeframe == "Weekly"),
        None,
    )
    if weekly is None:
        return

    arrays = candidate_arrays(analyses)
    run_dates = weekly.ohlc.index.to_numpy()
    positions = {
        timeframe: np.searchsorted(
            next(a.ohlc.index.to_numpy() for (tf, _, _), a in analyses.items() if tf == timeframe),
            run_dates,
            side="right",
        )
        - 1
        for timeframe in {key[0] for key in analyses}
    }

    seen: set[tuple[str, int]] = set()
    for run_index in range(len(run_dates)):
        for timeframe, length in scanner.HIERARCHY:
            if timeframe not in positions:
                continue
            latest = int(positions[timeframe][run_index])
            if latest < 1:
                continue

            best: tuple[str, dict[str, np.ndarray]] | None = None
            for average_type in scanner.AVERAGE_TYPES:
                data = arrays.get((timeframe, length, average_type))
                if data is None or data["direction"][latest] == 0:
                    continue
                if not (
                    np.isfinite(data["ma"][latest])
                    and np.isfinite(data["ma"][latest - 1])
                    and data["score"][latest] > 0
                ):
                    continue
                if best is None or data["score"][latest] > best[1]["score"][latest]:
                    best = (average_type, data)

            if best is None:
                continue
            if every_run_date or (timeframe, latest) not in seen:
                seen.add((timeframe, latest))
                yield Setup(
                    ticker=ticker,
                    run_date=pd.Timestamp(run_dates[run_index]),
                    timeframe=timeframe,
                    length=length,
                    average_type=best[0],
                    direction=int(best[1]["direction"][latest]),
                    position=latest,
                )
            break  # the first hierarchy step with a dominant level decides


def calculate_atr(ohlc: pd.DataFrame, length: int = ATR_LENGTH) -> np.ndarray:
    """Wilder's average true range. Each value only depends on earlier candles."""
    high, low, close = (ohlc[column] for column in ("High", "Low", "Close"))
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    return (
        true_range.ewm(alpha=1 / length, adjust=False, min_periods=length)
        .mean()
        .to_numpy(dtype=float)
    )


def simulate_pullback(
    setup: Setup,
    analyses: dict[tuple[str, int, str], scanner.CandidateAnalysis],
    entry_atr: float,
    stop_atr: float,
    max_candles: int,
    cost_pct: float,
    use_target: bool = True,
) -> Trade | None:
    """Simulate a pullback to the dominant average in the candle after `setup`.

    Long at dominant support (short at dominant resistance is the mirror image).
    Distances are in units of the average true range (ATR) of the signal's own
    timeframe, so a monthly stop sits further from the average than a weekly
    one, and a volatile stock gets more room than a quiet one:

    - Entry when the next candle trades down to `entry_atr` ATR above the
      average. A candle that opens below that level is entered at its open,
      unless it opens beyond the stop.
    - Stop `stop_atr` ATR below the average. It follows the average from candle
      to candle and only ever tightens.
    - Target is the nearest other average (20, 50 or 200, same SMA/EMA type)
      above the entry. With no average above, or with `use_target` off, there is
      no target and the trailed stop alone ends the trade.

    The average of the last completed candle is used for the entry and stop of
    the candle being traded. Returns None when the candle never reaches the
    entry level, or when the ATR is not known yet.
    """
    analysis = analyses[(setup.timeframe, setup.length, setup.average_type)]
    ohlc = analysis.ohlc
    known = setup.position
    entry_at = known + 1
    if entry_at >= len(ohlc):
        return None

    dates = ohlc.index
    open_, high, low, close = (
        ohlc[column].to_numpy(dtype=float) for column in ("Open", "High", "Low", "Close")
    )
    dominant = analysis.ma.to_numpy(dtype=float)
    atr = calculate_atr(ohlc)
    if not np.isfinite(atr[known]):
        return None
    long = setup.direction == SUPPORT
    sign = 1.0 if long else -1.0

    def beyond(index: int, units: float) -> float:
        """The average of candle `index` moved `units` ATR in the trade's direction."""
        return dominant[index] + sign * units * atr[index]

    entry_level = beyond(known, entry_atr)
    stop = initial_stop = beyond(known, -stop_atr)

    def reached(index: int, level: float, favorable: bool) -> bool:
        """Whether candle `index` traded at `level` on the profit or loss side."""
        if favorable == long:
            return bool(high[index] >= level)
        return bool(low[index] <= level)

    if not reached(entry_at, entry_level, favorable=False):
        return None
    opened_through = sign * (open_[entry_at] - entry_level) <= 0
    entry = open_[entry_at] if opened_through else entry_level
    if sign * (entry - stop) <= 0:
        return None  # opened beyond the stop: there was no pullback to trade
    risk = abs(entry - stop)

    target_length = None
    target_series = None
    for length in TARGET_LENGTHS if use_target else ():
        if length == setup.length or known + 1 < length:
            continue
        series = scanner.calculate_average(
            ohlc["Close"], length, setup.average_type
        ).to_numpy(dtype=float)
        value = series[known]
        if not np.isfinite(value) or sign * (value - entry) <= 0:
            continue
        if target_series is None or abs(value - entry) < abs(target_series[known] - entry):
            target_length, target_series = length, series

    exit_at, exit_price, reason = len(close) - 1, close[-1], "open"
    if reached(entry_at, stop, favorable=False):
        # Price fell through the entry and on to the stop within the same candle.
        exit_at, exit_price, reason = entry_at, stop, "stop"
    else:
        for k in range(entry_at + 1, len(close)):
            if np.isfinite(dominant[k - 1]) and np.isfinite(atr[k - 1]):
                level = beyond(k - 1, -stop_atr)
                stop = max(stop, level) if long else min(stop, level)
            if reached(k, stop, favorable=False):
                gapped = sign * (open_[k] - stop) <= 0
                reason = "stop" if stop == initial_stop else "trail"
                exit_at, exit_price = k, (open_[k] if gapped else stop)
                break
            if target_series is not None and reached(k, target_series[k - 1], favorable=True):
                target = target_series[k - 1]
                gapped = sign * (open_[k] - target) >= 0
                exit_at, exit_price, reason = k, (open_[k] if gapped else target), "target"
                break
            if k - entry_at >= max_candles:
                exit_at, exit_price, reason = k, close[k], "time"
                break

    return_pct = sign * (exit_price / entry - 1) * 100 - 2 * cost_pct
    r_multiple = sign * (exit_price - entry) / risk - 2 * cost_pct / 100 * entry / risk
    divergence = scanner.divergence_context(ohlc, known)

    return Trade(
        setup="pullback",
        ticker=setup.ticker,
        side="long" if long else "short",
        timeframe=setup.timeframe,
        length=setup.length,
        average_type=setup.average_type,
        bars_streak=int(analysis.bars_streak.iloc[known]),
        streak_touches=int(analysis.streak_touches.iloc[known]),
        faulty_candles=int(analysis.faulty_candles.iloc[known]),
        score=round(float(analysis.score.iloc[known]), 2),
        rsi_at_entry=divergence["rsi"],
        negative_divergence=divergence["negative_divergence"],
        positive_divergence=divergence["positive_divergence"],
        entry_date=pd.Timestamp(dates[entry_at]).date().isoformat(),
        entry_price=round(float(entry), 4),
        target=(
            "none"
            if target_series is None
            else f"{setup.timeframe} {target_length}{setup.average_type}"
        ),
        target_at_entry=(
            None if target_series is None else round(float(target_series[known]), 4)
        ),
        stop_at_entry=round(float(initial_stop), 4),
        exit_date=pd.Timestamp(dates[exit_at]).date().isoformat(),
        exit_price=round(float(exit_price), 4),
        exit_reason=reason,
        candles_held=exit_at - entry_at,
        return_pct=round(float(return_pct), 3),
        r_multiple=round(float(r_multiple), 3),
    )


def find_pullbacks(
    ticker: str,
    analyses: dict[tuple[str, int, str], scanner.CandidateAnalysis],
    entry_atr: float = 0.25,
    stop_atr: float = 0.75,
    max_candles: int = 104,
    cost_pct: float = 0.1,
    max_attempts: int = 3,
    attempt_window: int = 26,
    use_target: bool = True,
) -> list[Trade]:
    """Trade every pullback to the dominant level, one position per idea at a time.

    An idea is a timeframe and an average length: the SMA and the EMA of the
    same length are one idea, whichever of the two is dominant at the time.
    After `max_attempts` losing pullbacks in a row on one idea, no further
    attempt is made until `attempt_window` candles have passed since the last.
    """
    trades: list[Trade] = []
    busy_until: dict[tuple[str, int], int] = {}
    losses: dict[tuple[str, int], tuple[int, int]] = {}  # (count, last entry candle)
    for setup in iter_dominant_levels(ticker, analyses):
        level = (setup.timeframe, setup.length)
        entry_at = setup.position + 1
        if entry_at <= busy_until.get(level, -1):
            continue
        count, last_entry = losses.get(level, (0, -attempt_window))
        if count >= max_attempts:
            if entry_at - last_entry <= attempt_window:
                continue
            count = 0

        trade = simulate_pullback(
            setup, analyses, entry_atr, stop_atr, max_candles, cost_pct, use_target
        )
        if trade is None:
            continue
        trades.append(trade)
        busy_until[level] = entry_at + trade.candles_held
        losses[level] = (count + 1, entry_at) if trade.return_pct <= 0 else (0, entry_at)
    return trades


def find_signals(
    ticker: str,
    analyses: dict[tuple[str, int, str], scanner.CandidateAnalysis],
) -> list[Signal]:
    """Return each break once, at the first run date that reports it."""
    seen: set[tuple[str, int]] = set()
    signals: list[Signal] = []
    for signal in iter_run_signals(ticker, analyses):
        key = (signal.timeframe, signal.position)
        if key in seen:
            continue
        seen.add(key)
        signals.append(signal)
    return signals


def simulate_trade(
    signal: Signal,
    analyses: dict[tuple[str, int, str], scanner.CandidateAnalysis],
    stop_mode: str,
    max_candles: int,
    cost_pct: float,
    stop_buffer_pct: float,
    trail_after_r: float = 1.0,
) -> Trade | None:
    """Simulate one signal. Returns None when no target average exists."""
    analysis = analyses[(signal.timeframe, signal.length, signal.average_type)]
    ohlc = analysis.ohlc
    dates = ohlc.index
    open_, high, low, close = (
        ohlc[column].to_numpy(dtype=float) for column in ("Open", "High", "Low", "Close")
    )
    dominant = analysis.ma.to_numpy(dtype=float)
    entry_at = signal.position
    entry = close[entry_at]
    short = signal.signal_type == "break_below_dominant_support"
    sign = -1.0 if short else 1.0

    target_length = None
    target_series = None
    for length in TARGET_LENGTHS:
        if length <= signal.length or entry_at + 1 < length:
            continue
        series = scanner.calculate_average(
            ohlc["Close"], length, signal.average_type
        ).to_numpy(dtype=float)
        value = series[entry_at]
        if np.isfinite(value) and (value < entry if short else value > entry):
            target_length, target_series = length, series
            break
    if target_series is None:
        return None

    stop = None
    buffer = 1 + stop_buffer_pct / 100
    if stop_mode in ("candle", "trail"):
        stop = high[entry_at] * buffer if short else low[entry_at] / buffer
    initial_stop = stop
    risk = abs(stop - entry) if stop is not None else None
    trailing = False

    exit_at, exit_price, reason = len(close) - 1, close[-1], "open"
    for k in range(entry_at + 1, len(close)):
        target = target_series[k - 1]  # known before candle k starts
        if stop_mode == "trail" and risk:
            # Everything here uses candle k - 1, so the stop is set before candle k.
            if not trailing and sign * (close[k - 1] - entry) >= trail_after_r * risk:
                trailing = True
            if trailing and np.isfinite(dominant[k - 1]):
                level = dominant[k - 1] * buffer if short else dominant[k - 1] / buffer
                stop = min(stop, level) if short else max(stop, level)
        if stop is not None and (high[k] >= stop if short else low[k] <= stop):
            gapped = open_[k] >= stop if short else open_[k] <= stop
            reason = "stop" if stop == initial_stop else "trail"
            exit_at, exit_price = k, (open_[k] if gapped else stop)
            break
        if low[k] <= target if short else high[k] >= target:
            gapped = open_[k] <= target if short else open_[k] >= target
            exit_at, exit_price, reason = k, (open_[k] if gapped else target), "target"
            break
        if stop_mode == "reclaim" and (
            close[k] > dominant[k] if short else close[k] < dominant[k]
        ):
            exit_at, exit_price, reason = k, close[k], "stop"
            break
        if k - entry_at >= max_candles:
            exit_at, exit_price, reason = k, close[k], "time"
            break

    return_pct = sign * (exit_price / entry - 1) * 100 - 2 * cost_pct
    r_multiple = None
    if risk:
        r_multiple = sign * (exit_price - entry) / risk - 2 * cost_pct / 100 * entry / risk

    divergence = scanner.divergence_context(ohlc, entry_at)

    return Trade(
        setup="break",
        ticker=signal.ticker,
        side="short" if short else "long",
        timeframe=signal.timeframe,
        length=signal.length,
        average_type=signal.average_type,
        bars_streak=signal.bars_streak,
        streak_touches=signal.streak_touches,
        faulty_candles=signal.faulty_candles,
        score=round(signal.score, 2),
        rsi_at_entry=divergence["rsi"],
        negative_divergence=divergence["negative_divergence"],
        positive_divergence=divergence["positive_divergence"],
        entry_date=pd.Timestamp(dates[entry_at]).date().isoformat(),
        entry_price=round(float(entry), 4),
        target=f"{signal.timeframe} {target_length}{signal.average_type}",
        target_at_entry=round(float(target_series[entry_at]), 4),
        stop_at_entry=None if initial_stop is None else round(float(initial_stop), 4),
        exit_date=pd.Timestamp(dates[exit_at]).date().isoformat(),
        exit_price=round(float(exit_price), 4),
        exit_reason=reason,
        candles_held=exit_at - entry_at,
        return_pct=round(float(return_pct), 3),
        r_multiple=None if r_multiple is None else round(float(r_multiple), 3),
    )


def clean_history(
    daily: pd.DataFrame,
    max_jump: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Drop the history before the last implausible one-day price jump.

    A close more than `max_jump` times the previous close, or less than
    1 / `max_jump` of it, is treated as a data error, such as a share split
    applied to only part of the history. Everything up to and including that
    day is dropped. Returns the remaining data and the jumps found.
    """
    close = daily["Close"]
    ratio = close / close.shift(1)
    bad = ratio[(ratio > max_jump) | (ratio < 1 / max_jump)]
    flags = [
        {
            "date": pd.Timestamp(date).date().isoformat(),
            "previous_close": round(float(close.shift(1).loc[date]), 6),
            "close": round(float(close.loc[date]), 6),
            "ratio": round(float(value), 3),
        }
        for date, value in bad.items()
    ]
    if bad.empty:
        return daily, flags
    return daily.loc[daily.index > bad.index[-1]], flags


def simulate_retracement(
    signal: Signal,
    analyses: dict[tuple[str, int, str], scanner.CandidateAnalysis],
    max_wait: int,
    max_attempts: int,
    max_candles: int,
    cost_pct: float,
) -> list[Trade]:
    """Simulate JDB's retracement short after a break below dominant support.

    The retracement is drawn on the signal's own timeframe:

    - 100% is the highest high of the dominant streak that just broke.
    - 0% is the lowest low since that top. It is redrawn whenever price makes a
      new low, using only completed candles.
    - Entry: short when a candle trades up to exactly 50%, coming from below:
      the previous candle closed under 50%. A candle that opens above 50% (but
      below the stop) is entered at its open, as a resting order would be.
    - Stop: one fifth of the way from 61.8% to 100%, about 69.4%, fixed at entry.
    - Target: the first 200-period average reached, SMA or EMA, whichever is
      higher. An attempt is only made while that target is below the entry.

    The setup is watched for `max_wait` candles after the break. It ends early
    when price closes above the top, or when price reaches the 200-period
    average before an entry. A stopped-out attempt may be followed by another
    one, up to `max_attempts` in total, but only after price has made a new
    low, so the next attempt is drawn from a lower 0% level.
    """
    if signal.signal_type != "break_below_dominant_support":
        return []
    analysis = analyses[(signal.timeframe, signal.length, signal.average_type)]
    ohlc = analysis.ohlc
    dates = ohlc.index
    open_, high, low, close = (
        ohlc[column].to_numpy(dtype=float) for column in ("Open", "High", "Low", "Close")
    )
    if len(close) < 200:
        return []
    sma200 = scanner.calculate_average(ohlc["Close"], 200, "SMA").to_numpy(dtype=float)
    ema200 = scanner.calculate_average(ohlc["Close"], 200, "EMA").to_numpy(dtype=float, copy=True)
    ema200[:199] = np.nan  # an EMA needs a full window before it means anything
    target_line = np.fmax(sma200, ema200)

    broke = signal.position
    first = max(0, broke - signal.bars_streak)
    top = float(np.max(high[first : broke + 1]))
    divergence = scanner.divergence_context(ohlc, broke)

    trades: list[Trade] = []
    swing_low = float(np.min(low[int(np.argmax(high[first : broke + 1])) + first : broke + 1]))
    needs_new_low_below: float | None = None  # set after a stopped-out attempt
    k = broke + 1
    last_watch = min(len(close) - 1, broke + max_wait)
    while k <= last_watch and len(trades) < max_attempts:
        # Levels for candle k, from candles up to k - 1 only.
        span = top - swing_low
        target = target_line[k - 1]
        if span <= 0 or not np.isfinite(target):
            swing_low = min(swing_low, low[k])
            k += 1
            continue
        entry_level = swing_low + 0.5 * span
        stop_level = swing_low + (0.618 + 0.2 * (1 - 0.618)) * span
        if target >= entry_level:
            # No room between the 50% level and the 200-period average.
            if low[k] <= target and not trades:
                break  # price reached the target first: the move is over
            swing_low = min(swing_low, low[k])
            k += 1
            continue

        ready = needs_new_low_below is None or swing_low < needs_new_low_below
        if ready and close[k - 1] < entry_level and high[k] >= entry_level and open_[k] < stop_level:
            entry = max(open_[k], entry_level)
            risk = stop_level - entry
            entry_at = k
            exit_at, exit_price, reason = len(close) - 1, close[-1], "open"
            if high[k] >= stop_level:
                # Rallied through 50% and on to the stop inside the entry candle.
                exit_at, exit_price, reason = k, stop_level, "stop"
            else:
                for j in range(k + 1, len(close)):
                    level = target_line[j - 1]
                    if high[j] >= stop_level:
                        gapped = open_[j] >= stop_level
                        exit_at, exit_price = j, (open_[j] if gapped else stop_level)
                        reason = "stop"
                        break
                    if np.isfinite(level) and low[j] <= level:
                        gapped = open_[j] <= level
                        exit_at, exit_price = j, (open_[j] if gapped else level)
                        reason = "target"
                        break
                    if j - entry_at >= max_candles:
                        exit_at, exit_price, reason = j, close[j], "time"
                        break

            return_pct = -(exit_price / entry - 1) * 100 - 2 * cost_pct
            r_multiple = (entry - exit_price) / risk - 2 * cost_pct / 100 * entry / risk
            trades.append(
                Trade(
                    setup="retracement",
                    ticker=signal.ticker,
                    side="short",
                    timeframe=signal.timeframe,
                    length=signal.length,
                    average_type=signal.average_type,
                    bars_streak=signal.bars_streak,
                    streak_touches=signal.streak_touches,
                    faulty_candles=signal.faulty_candles,
                    score=round(signal.score, 2),
                    rsi_at_entry=divergence["rsi"],
                    negative_divergence=divergence["negative_divergence"],
                    positive_divergence=divergence["positive_divergence"],
                    entry_date=pd.Timestamp(dates[entry_at]).date().isoformat(),
                    entry_price=round(float(entry), 4),
                    target=f"{signal.timeframe} 200 (first of SMA/EMA)",
                    target_at_entry=round(float(target), 4),
                    stop_at_entry=round(float(stop_level), 4),
                    exit_date=pd.Timestamp(dates[exit_at]).date().isoformat(),
                    exit_price=round(float(exit_price), 4),
                    exit_reason=reason,
                    candles_held=exit_at - entry_at,
                    return_pct=round(float(return_pct), 3),
                    r_multiple=round(float(r_multiple), 3),
                )
            )
            if reason != "stop":
                break  # target, time or still open: the idea has played out
            needs_new_low_below = swing_low
            k = exit_at + 1
            last_watch = min(len(close) - 1, max(last_watch, k + max_wait // 2))
            continue

        if close[k] > top:
            break  # a new high: the bearish idea is invalid
        swing_low = min(swing_low, low[k])
        k += 1
    return trades


def random_baseline(
    trades: list[Trade],
    analyses: dict[tuple[str, int, str], scanner.CandidateAnalysis],
    per_trade: int,
    max_candles: int,
    cost_pct: float,
) -> list[dict[str, Any]]:
    """Trade the same geometry as each signal at random dates, as a control.

    For every closed break and retracement trade, `per_trade` trades are made in
    the same ticker, timeframe and direction, entered at the close of a random
    candle, with the stop and target at the same percentage distance from the
    entry as the real trade had. Stop and target stay fixed. If the signal adds
    nothing, the real trades and these random ones should score about the same,
    whatever the stock universe's own drift is.
    """
    rows: list[dict[str, Any]] = []
    for trade in trades:
        if trade.setup not in ("break", "retracement") or trade.exit_reason == "open":
            continue
        if trade.stop_at_entry is None or trade.target_at_entry is None:
            continue
        ohlc = analyses[(trade.timeframe, trade.length, trade.average_type)].ohlc
        open_, high, low, close = (
            ohlc[column].to_numpy(dtype=float) for column in ("Open", "High", "Low", "Close")
        )
        if len(close) < 260:
            continue
        short = trade.side == "short"
        sign = -1.0 if short else 1.0
        stop_ratio = trade.stop_at_entry / trade.entry_price
        target_ratio = trade.target_at_entry / trade.entry_price
        seed = zlib.crc32(f"{trade.ticker}|{trade.setup}|{trade.entry_date}".encode())
        rng = np.random.default_rng(seed)
        for position in rng.integers(200, len(close) - 1, size=per_trade):
            entry = close[position]
            stop, target = entry * stop_ratio, entry * target_ratio
            risk = abs(entry - stop)
            if risk <= 0:
                continue
            exit_price, reason = None, None
            for k in range(position + 1, len(close)):
                if high[k] >= stop if short else low[k] <= stop:
                    gapped = open_[k] >= stop if short else open_[k] <= stop
                    exit_price, reason = (open_[k] if gapped else stop), "stop"
                    break
                if low[k] <= target if short else high[k] >= target:
                    gapped = open_[k] <= target if short else open_[k] >= target
                    exit_price, reason = (open_[k] if gapped else target), "target"
                    break
                if k - position >= max_candles:
                    exit_price, reason = close[k], "time"
                    break
            if exit_price is None:
                continue  # still open at the end of the data
            rows.append(
                {
                    "setup": trade.setup,
                    "side": trade.side,
                    "timeframe": trade.timeframe,
                    "length": trade.length,
                    "ticker": trade.ticker,
                    "entry_date": pd.Timestamp(ohlc.index[position]).date().isoformat(),
                    "exit_reason": reason,
                    "return_pct": round(
                        float(sign * (exit_price / entry - 1) * 100 - 2 * cost_pct), 3
                    ),
                    "r_multiple": round(
                        float(sign * (exit_price - entry) / risk - 2 * cost_pct / 100 * entry / risk),
                        3,
                    ),
                }
            )
    return rows


def baseline_summary(
    trades: list[Trade],
    baseline: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compare real break and retracement trades with their random-date controls."""
    result: dict[str, dict[str, Any]] = {}
    for setup in ("break", "retracement"):
        for side in ("long", "short"):
            real = [
                trade.r_multiple
                for trade in trades
                if trade.setup == setup
                and trade.side == side
                and trade.exit_reason != "open"
                and trade.r_multiple is not None
            ]
            control = [
                row["r_multiple"]
                for row in baseline
                if row["setup"] == setup and row["side"] == side
            ]
            if not real or not control:
                continue
            result[f"{setup} {side}"] = {
                "signal_trades": len(real),
                "signal_avg_r": round(float(np.mean(real)), 3),
                "signal_win_rate_pct": round(float(np.mean(np.array(real) > 0) * 100), 1),
                "random_trades": len(control),
                "random_avg_r": round(float(np.mean(control)), 3),
                "random_win_rate_pct": round(float(np.mean(np.array(control) > 0) * 100), 1),
            }
    return result


def backtest_ticker(
    ticker: str,
    daily: pd.DataFrame,
    stop_mode: str = "candle",
    max_candles: int = 104,
    cost_pct: float = 0.1,
    stop_buffer_pct: float = 0.5,
    start: pd.Timestamp | None = None,
    trail_after_r: float = 1.0,
    setups: tuple[str, ...] = ("break",),
    pullback_entry_atr: float = 0.25,
    pullback_stop_atr: float = 0.75,
    retrace_max_wait: int = 52,
    retrace_attempts: int = 3,
    pullback_target: str = "next",
) -> tuple[list[Trade], int]:
    """Backtest one ticker. Returns its trades and the number of untradable breaks."""
    analyses = scanner.build_candidate_analyses(daily)
    trades: list[Trade] = []
    skipped = 0
    if "retracement" in setups:
        busy_until: dict[str, pd.Timestamp] = {}
        for signal in find_signals(ticker, analyses):
            if start is not None and signal.run_date < start:
                continue
            break_date = analyses[
                (signal.timeframe, signal.length, signal.average_type)
            ].ohlc.index[signal.position]
            if break_date <= busy_until.get(signal.timeframe, pd.Timestamp.min):
                continue  # one retracement idea per timeframe at a time
            idea = simulate_retracement(
                signal, analyses, retrace_max_wait, retrace_attempts, max_candles, cost_pct
            )
            if idea:
                trades.extend(idea)
                busy_until[signal.timeframe] = pd.Timestamp(idea[-1].exit_date)
    if "pullback" in setups:
        trades.extend(
            trade
            for trade in find_pullbacks(
                ticker,
                analyses,
                entry_atr=pullback_entry_atr,
                stop_atr=pullback_stop_atr,
                max_candles=max_candles,
                cost_pct=cost_pct,
                use_target=pullback_target == "next",
            )
            if start is None or pd.Timestamp(trade.entry_date) >= start
        )
    for signal in find_signals(ticker, analyses) if "break" in setups else []:
        if start is not None and signal.run_date < start:
            continue
        trade = simulate_trade(
            signal, analyses, stop_mode, max_candles, cost_pct, stop_buffer_pct, trail_after_r
        )
        if trade is None:
            skipped += 1
        else:
            trades.append(trade)
    return trades, skipped


def longest_losing_streak(returns: list[float]) -> int:
    """Return the longest run of consecutive losing trades."""
    longest = current = 0
    for value in returns:
        current = current + 1 if value <= 0 else 0
        longest = max(longest, current)
    return longest


def summarize(trades: list[Trade]) -> dict[str, Any]:
    """Summarize closed trades. Open trades are counted but not scored."""
    closed = sorted(
        (trade for trade in trades if trade.exit_reason != "open"),
        key=lambda trade: trade.exit_date,
    )
    returns = [trade.return_pct for trade in closed]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    r_values = [trade.r_multiple for trade in closed if trade.r_multiple is not None]

    def mean(values: list[float]) -> float | None:
        return round(float(np.mean(values)), 3) if values else None

    return {
        "closed_trades": len(closed),
        "open_trades": len(trades) - len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "avg_win_pct": mean(wins),
        "avg_loss_pct": mean(losses),
        "expectancy_pct": mean(returns),
        "median_return_pct": round(float(np.median(returns)), 3) if returns else None,
        "avg_r": mean(r_values),
        "median_candles_held": (
            int(np.median([trade.candles_held for trade in closed])) if closed else None
        ),
        "longest_losing_streak": longest_losing_streak(returns),
        "exits": {
            reason: sum(1 for trade in closed if trade.exit_reason == reason)
            for reason in ("target", "stop", "trail", "time")
        },
    }


def confirming_divergence(trade: Trade) -> bool | None:
    """Divergence in the trade's direction: negative for shorts, positive for longs."""
    return trade.negative_divergence if trade.side == "short" else trade.positive_divergence


def grouped_summary(trades: list[Trade]) -> dict[str, dict[str, Any]]:
    """Summaries overall and per setup: by side, by divergence, by dominant level."""
    groups: dict[str, list[Trade]] = {"all": trades}
    for trade in trades:
        side = f"{trade.setup} {trade.side}"
        groups.setdefault(side, []).append(trade)
        confirmed = confirming_divergence(trade)
        if trade.setup == "break" and confirmed is not None:
            label = "with" if confirmed else "without"
            groups.setdefault(f"{side} {label} divergence", []).append(trade)
        groups.setdefault(f"{side} / {trade.timeframe} {trade.length}", []).append(trade)

    def order(name: str) -> tuple[str, int, str]:
        rank = 2 if " / " in name else 1 if name.endswith("divergence") else 0
        return (" ".join(name.split(" ")[:2]), rank, name)

    ordered = ["all"] + sorted((key for key in groups if key != "all"), key=order)
    return {key: summarize(groups[key]) for key in ordered}


def write_outputs(
    trades: list[Trade],
    summary: dict[str, Any],
    out_dir: Path,
) -> None:
    """Write the trade list and the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(Trade.__dataclass_fields__)
    with (out_dir / "backtest_trades.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(trade) for trade in trades)
    with (out_dir / "backtest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def print_summary(summary: dict[str, Any]) -> None:
    """Print the grouped summary as a table."""
    header = (
        f"{'group':<36}{'trades':>7}{'win%':>7}{'avg win':>9}{'avg loss':>10}"
        f"{'expect%':>9}{'avg R':>7}{'held':>6}{'streak':>8}"
    )
    print(header)
    print("-" * len(header))

    def cell(value: Any, width: int) -> str:
        return f"{'-' if value is None else value:>{width}}"

    for name, row in summary["groups"].items():
        print(
            f"{name:<36}{row['closed_trades']:>7}"
            + cell(row["win_rate_pct"], 7)
            + cell(row["avg_win_pct"], 9)
            + cell(row["avg_loss_pct"], 10)
            + cell(row["expectancy_pct"], 9)
            + cell(row["avg_r"], 7)
            + cell(row["median_candles_held"], 6)
            + cell(row["longest_losing_streak"], 8)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tickers-file", type=Path, default=scanner.TICKERS_FILE)
    parser.add_argument("--start", help="ignore signals before this date (YYYY-MM-DD)")
    parser.add_argument("--stop", choices=("candle", "reclaim", "trail"), default="candle")
    parser.add_argument(
        "--trail-after-r",
        type=float,
        default=1.0,
        help="with --stop trail: profit, in units of initial risk, before the stop trails",
    )
    parser.add_argument(
        "--setups",
        default="break,pullback,retracement",
        help="comma-separated: break (close through the dominant average), "
        "pullback (return to the dominant average), "
        "retracement (short at 50%% after a break below dominant support)",
    )
    parser.add_argument(
        "--pullback-target",
        choices=("next", "none"),
        default="next",
        help="next: exit a pullback at the nearest other average; "
        "none: no target, ride the trend with the trailed stop",
    )
    parser.add_argument(
        "--retrace-max-wait",
        type=int,
        default=52,
        help="candles after a break during which the 50%% retracement is watched",
    )
    parser.add_argument(
        "--retrace-attempts",
        type=int,
        default=3,
        help="maximum retracement shorts per break",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=0,
        help="random-date control trades per break and retracement trade (0: off)",
    )
    parser.add_argument(
        "--max-daily-jump",
        type=float,
        default=4.0,
        help="a one-day price change beyond this factor is treated as a data error",
    )
    parser.add_argument(
        "--pullback-entry-atr",
        type=float,
        default=0.25,
        help="pullback entry, in ATR on the trend side of the dominant average",
    )
    parser.add_argument(
        "--pullback-stop-atr",
        type=float,
        default=0.75,
        help="pullback stop, in ATR beyond the dominant average",
    )
    parser.add_argument("--stop-buffer-pct", type=float, default=0.5)
    parser.add_argument("--max-candles", type=int, default=104)
    parser.add_argument("--cost-pct", type=float, default=0.1, help="cost per side, in percent")
    parser.add_argument("--out", type=Path, default=scanner.RESULTS_DIR / "backtest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start) if args.start else None
    setups = tuple(name.strip() for name in args.setups.split(",") if name.strip())
    unknown = set(setups) - {"break", "pullback", "retracement"}
    if unknown:
        raise SystemExit(f"Unknown setup: {', '.join(sorted(unknown))}")
    tickers = scanner.load_tickers(args.tickers_file)
    price_data, errors = scanner.download_price_data_bulk(tickers)

    trades: list[Trade] = []
    skipped = 0
    data_flags: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    for ticker in tickers:
        daily = price_data.get(ticker)
        if daily is None:
            continue
        daily, flags = clean_history(daily, args.max_daily_jump)
        data_flags.extend({"ticker": ticker, **flag} for flag in flags)
        if len(daily) < 260:
            errors[ticker] = "too little clean price history"
            continue
        try:
            ticker_trades, ticker_skipped = backtest_ticker(
                ticker,
                daily,
                stop_mode=args.stop,
                max_candles=args.max_candles,
                cost_pct=args.cost_pct,
                stop_buffer_pct=args.stop_buffer_pct,
                start=start,
                trail_after_r=args.trail_after_r,
                setups=setups,
                pullback_entry_atr=args.pullback_entry_atr,
                pullback_stop_atr=args.pullback_stop_atr,
                retrace_max_wait=args.retrace_max_wait,
                retrace_attempts=args.retrace_attempts,
                pullback_target=args.pullback_target,
            )
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the run.
            errors[ticker] = str(exc)
            continue
        trades.extend(ticker_trades)
        skipped += ticker_skipped
        if args.baseline > 0:
            baseline.extend(
                random_baseline(
                    ticker_trades,
                    scanner.build_candidate_analyses(daily),
                    args.baseline,
                    args.max_candles,
                    args.cost_pct,
                )
            )

    summary = {
        "settings": {
            "setups": list(setups),
            "pullback_entry_atr": args.pullback_entry_atr,
            "pullback_stop_atr": args.pullback_stop_atr,
            "pullback_target": args.pullback_target,
            "retrace_max_wait": args.retrace_max_wait,
            "retrace_attempts": args.retrace_attempts,
            "max_daily_jump": args.max_daily_jump,
            "stop": args.stop,
            "stop_buffer_pct": args.stop_buffer_pct,
            "trail_after_r": args.trail_after_r if args.stop == "trail" else None,
            "max_candles": args.max_candles,
            "cost_pct_per_side": args.cost_pct,
            "start": args.start,
            "prices": "dividend adjusted" if scanner.YFINANCE_AUTO_ADJUST else "unadjusted",
            "min_touches": scanner.MIN_TOUCHES,
            "faulty_candles_allowed": scanner.FAULTY_CANDLES_ALLOWED,
            "faulty_candle_max_pct": scanner.FAULTY_CANDLE_MAX_PCT,
            "dominant_lengths": list(scanner.DOMINANT_LENGTHS),
        },
        "tickers": len(tickers),
        "tickers_with_errors": len(errors),
        "tickers_cut_for_data_errors": sorted({flag["ticker"] for flag in data_flags}),
        "breaks_without_target": skipped,
        "groups": grouped_summary(trades),
    }
    if args.baseline > 0:
        summary["random_baseline"] = baseline_summary(trades, baseline)
    write_outputs(trades, summary, args.out)
    with (args.out / "data_flags.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["ticker", "date", "previous_close", "close", "ratio"]
        )
        writer.writeheader()
        writer.writerows(data_flags)
    if baseline:
        with (args.out / "baseline_trades.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(baseline[0]))
            writer.writeheader()
            writer.writerows(baseline)
        print("\nSignal against random dates with the same stop and target distances:")
        for name, row in summary["random_baseline"].items():
            print(
                f"{name:<20} signal {row['signal_avg_r']:+.3f} R ({row['signal_trades']}), "
                f"random {row['random_avg_r']:+.3f} R ({row['random_trades']})"
            )
    if errors:
        with (args.out / "errors.json").open("w", encoding="utf-8") as handle:
            json.dump(errors, handle, indent=2)
    print_summary(summary)
    print(
        f"\n{len(trades)} trades from {len(tickers) - len(errors)} tickers; "
        f"{skipped} breaks had no average left to target; {len(errors)} tickers failed."

    )
    print(f"Wrote {args.out / 'backtest_trades.csv'} and {args.out / 'backtest_summary.json'}")


if __name__ == "__main__":
    main()
