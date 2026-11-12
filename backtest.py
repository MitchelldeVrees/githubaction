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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import scanner

TARGET_LENGTHS = (20, 50, 200)
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
    score: float


@dataclass
class Trade:
    """Outcome of trading one signal."""

    ticker: str
    side: str
    timeframe: str
    length: int
    average_type: str
    bars_streak: int
    score: float
    rsi_at_entry: float | None
    negative_divergence: bool | None
    positive_divergence: bool | None
    entry_date: str
    entry_price: float
    target: str
    target_at_entry: float
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
                    score=float(data["score"][previous]),
                )
            break  # the first hierarchy step with a dominant level decides


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
        ticker=signal.ticker,
        side="short" if short else "long",
        timeframe=signal.timeframe,
        length=signal.length,
        average_type=signal.average_type,
        bars_streak=signal.bars_streak,
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


def backtest_ticker(
    ticker: str,
    daily: pd.DataFrame,
    stop_mode: str = "candle",
    max_candles: int = 104,
    cost_pct: float = 0.1,
    stop_buffer_pct: float = 0.5,
    start: pd.Timestamp | None = None,
    trail_after_r: float = 1.0,
) -> tuple[list[Trade], int]:
    """Backtest one ticker. Returns its trades and the number of untradable breaks."""
    analyses = scanner.build_candidate_analyses(daily)
    trades: list[Trade] = []
    skipped = 0
    for signal in find_signals(ticker, analyses):
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
    """Summaries overall, per side, per divergence, and per dominant level."""
    groups: dict[str, list[Trade]] = {"all": trades}
    for trade in trades:
        groups.setdefault(trade.side, []).append(trade)
        confirmed = confirming_divergence(trade)
        if confirmed is not None:
            label = "with" if confirmed else "without"
            groups.setdefault(f"{trade.side} {label} divergence", []).append(trade)
        level = f"{trade.side} / {trade.timeframe} {trade.length}"
        groups.setdefault(level, []).append(trade)

    def order(name: str) -> tuple[str, int, str]:
        rank = 2 if " / " in name else 1 if name.endswith("divergence") else 0
        return (name.split(" ")[0], rank, name)

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
        f"{'group':<26}{'trades':>7}{'win%':>7}{'avg win':>9}{'avg loss':>10}"
        f"{'expect%':>9}{'avg R':>7}{'held':>6}{'streak':>8}"
    )
    print(header)
    print("-" * len(header))

    def cell(value: Any, width: int) -> str:
        return f"{'-' if value is None else value:>{width}}"

    for name, row in summary["groups"].items():
        print(
            f"{name:<26}{row['closed_trades']:>7}"
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
    parser.add_argument("--stop-buffer-pct", type=float, default=0.5)
    parser.add_argument("--max-candles", type=int, default=104)
    parser.add_argument("--cost-pct", type=float, default=0.1, help="cost per side, in percent")
    parser.add_argument("--out", type=Path, default=scanner.RESULTS_DIR / "backtest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start) if args.start else None
    tickers = scanner.load_tickers(args.tickers_file)
    price_data, errors = scanner.download_price_data_bulk(tickers)

    trades: list[Trade] = []
    skipped = 0
    for ticker in tickers:
        daily = price_data.get(ticker)
        if daily is None:
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
            )
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the run.
            errors[ticker] = str(exc)
            continue
        trades.extend(ticker_trades)
        skipped += ticker_skipped

    summary = {
        "settings": {
            "stop": args.stop,
            "stop_buffer_pct": args.stop_buffer_pct,
            "trail_after_r": args.trail_after_r if args.stop == "trail" else None,
            "max_candles": args.max_candles,
            "cost_pct_per_side": args.cost_pct,
            "start": args.start,
            "prices": "dividend adjusted" if scanner.YFINANCE_AUTO_ADJUST else "unadjusted",
        },
        "tickers": len(tickers),
        "tickers_with_errors": len(errors),
        "breaks_without_target": skipped,
        "groups": grouped_summary(trades),
    }
    write_outputs(trades, summary, args.out)
    print_summary(summary)
    print(
        f"\n{len(trades)} trades from {len(tickers) - len(errors)} tickers; "
        f"{skipped} breaks had no average left to target; {len(errors)} tickers failed."
    )
    print(f"Wrote {args.out / 'backtest_trades.csv'} and {args.out / 'backtest_summary.json'}")


if __name__ == "__main__":
    main()
