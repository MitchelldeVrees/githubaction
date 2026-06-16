"""Dominant SMA/EMA stock scanner.

This script translates the supplied TradingView Pine Script dominance logic
into a GitHub Actions friendly Python scanner. It downloads daily Yahoo Finance
data, resamples it into higher-timeframe candles, selects the dominant level by
the requested hierarchy, writes CSV/JSON results, and optionally sends Telegram
alerts.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_str(name: str, default: str) -> str:
    """Read a string environment variable, treating an empty value as missing."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable."""
    return int(env_str(name, str(default)))


def env_float(name: str, default: float) -> float:
    """Read a float environment variable."""
    return float(env_str(name, str(default)))


def env_path(name: str, default: str) -> Path:
    """Read a path environment variable relative to this script by default."""
    raw_path = Path(env_str(name, default))
    return raw_path if raw_path.is_absolute() else BASE_DIR / raw_path


# Strategy parameters. These mirror the Pine Script defaults and can be
# overridden from GitHub Actions variables or a local shell.
BULL_BARS_REQUIRED = env_int("BULL_BARS_REQUIRED", 70)
BEAR_BARS_REQUIRED = env_int("BEAR_BARS_REQUIRED", 40)
TOUCH_DISTANCE_PCT = env_float("TOUCH_DISTANCE_PCT", 1.5)
BOUNCE_LOOKBACK = env_int("BOUNCE_LOOKBACK", 150)
BOUNCE_WEIGHT = env_int("BOUNCE_WEIGHT", 10)

SEND_NO_SIGNAL_MESSAGE = env_bool("SEND_NO_SIGNAL_MESSAGE", False)
TICKERS_FILE = env_path("TICKERS_FILE", "tickers.txt")
RESULTS_DIR = env_path("RESULTS_DIR", "results")

# yfinance with auto_adjust=True returns split/dividend adjusted OHLC values.
YFINANCE_PERIOD = env_str("YFINANCE_PERIOD", "max")
YFINANCE_AUTO_ADJUST = env_bool("YFINANCE_AUTO_ADJUST", True)


TIMEFRAME_RULES: dict[str, str] = {
    "Weekly": "W-FRI",
    "Biweekly": "2W-FRI",
    "Monthly": "ME",
    "Quarterly": "QE",
}

HIERARCHY: list[tuple[str, int]] = [
    ("Weekly", 20),
    ("Weekly", 50),
    ("Biweekly", 20),
    ("Biweekly", 50),
    ("Monthly", 20),
    ("Monthly", 50),
    ("Quarterly", 20),
    ("Quarterly", 50),
]

AVERAGE_TYPES = ("SMA", "EMA")
OHLC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

RESULT_FIELDS = [
    "ticker",
    "status",
    "signal_type",
    "dominant_timeframe",
    "dominant_length",
    "dominant_average_type",
    "dominant_direction",
    "latest_close",
    "previous_close",
    "dominant_value_latest",
    "dominant_value_previous",
    "score",
    "bounce_count",
    "bars_streak",
    "message",
    "error",
]


@dataclass(frozen=True)
class CandidateAnalysis:
    """Full candidate series for one timeframe/length/average type."""

    timeframe: str
    length: int
    average_type: str
    ohlc: pd.DataFrame
    ma: pd.Series
    direction: pd.Series
    score: pd.Series
    touch_count: pd.Series
    bars_streak: pd.Series


@dataclass(frozen=True)
class DominantSnapshot:
    """Selected dominant candidate at a specific candle."""

    ticker: str
    timeframe: str
    length: int
    average_type: str
    direction: str
    score: float
    touch_count: int
    bars_streak: int
    latest_close: float
    previous_close: float
    dominant_value_latest: float
    dominant_value_previous: float


def load_tickers(path: Path = TICKERS_FILE) -> list[str]:
    """Load tickers from a text file, ignoring blank lines and comments."""
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    tickers: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tickers.append(line.upper())
    return tickers


def normalize_yfinance_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a single-ticker OHLCV DataFrame from yfinance output."""
    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data = data.copy()
            data.columns = data.columns.get_level_values(0)

    data = data.rename(columns={column: str(column).title() for column in data.columns})
    missing = [column for column in OHLC_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {', '.join(missing)}")

    data = data[OHLC_COLUMNS].dropna(subset=["Open", "High", "Low", "Close"])
    data = data.sort_index()
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_convert(None)
    return data


def download_price_data(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV data for one Yahoo Finance ticker."""
    data = yf.download(
        ticker,
        period=YFINANCE_PERIOD,
        interval="1d",
        auto_adjust=YFINANCE_AUTO_ADJUST,
        progress=False,
        threads=False,
    )
    data = normalize_yfinance_columns(data, ticker)
    if data.empty:
        raise ValueError("Yahoo Finance returned no OHLC data")
    return data


def resample_ohlc(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample daily data into completed higher-timeframe OHLCV candles."""
    aggregation = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    resampled = daily.resample(rule, label="right", closed="right").agg(aggregation)
    resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])
    resampled["Volume"] = resampled["Volume"].fillna(0)

    # Avoid lookahead bias: the active higher-timeframe candle is not complete
    # until its calendar period end has arrived. GitHub Actions runs after the
    # close, so a candle whose period ends today can be used if Yahoo has data.
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    candle_end_dates = pd.to_datetime(resampled.index).tz_localize(None).normalize()
    return resampled.loc[candle_end_dates <= today]


def calculate_average(close: pd.Series, length: int, average_type: str) -> pd.Series:
    """Calculate an SMA or EMA series."""
    if average_type == "SMA":
        return close.rolling(window=length, min_periods=length).mean()
    if average_type == "EMA":
        return close.ewm(span=length, adjust=False, min_periods=1).mean()
    raise ValueError(f"Unsupported average type: {average_type}")


def bars_since_true(condition: pd.Series) -> pd.Series:
    """Mirror Pine Script ta.barssince behavior for latest-bar streak checks."""
    values: list[int] = []
    last_true_index: int | None = None

    for index, value in enumerate(condition.fillna(False).astype(bool).tolist()):
        if value:
            last_true_index = index
            values.append(0)
        elif last_true_index is None:
            values.append(index + 1)
        else:
            values.append(index - last_true_index)

    return pd.Series(values, index=condition.index, dtype="int64")


def analyze_candidate(
    ohlc: pd.DataFrame,
    timeframe: str,
    length: int,
    average_type: str,
) -> CandidateAnalysis:
    """Analyze one MA/EMA candidate using the Pine dominance rules."""
    close = ohlc["Close"]
    ma = calculate_average(close, length, average_type)

    bars_above_streak = bars_since_true(close <= ma)
    bars_below_streak = bars_since_true(close >= ma)

    bullish_bounce = (ohlc["Low"] <= ma * (1 + TOUCH_DISTANCE_PCT / 100)) & (close > ma)
    bearish_reject = (ohlc["High"] >= ma * (1 - TOUCH_DISTANCE_PCT / 100)) & (close < ma)

    bull_bounces = bullish_bounce.astype(int).rolling(
        BOUNCE_LOOKBACK, min_periods=1
    ).sum()
    bear_rejects = bearish_reject.astype(int).rolling(
        BOUNCE_LOOKBACK, min_periods=1
    ).sum()

    support_mask = (bars_above_streak >= BULL_BARS_REQUIRED) & ma.notna()
    resistance_mask = (
        (bars_below_streak >= BEAR_BARS_REQUIRED) & ma.notna() & ~support_mask
    )

    direction = pd.Series(0, index=ohlc.index, dtype="int64")
    direction.loc[support_mask] = 1
    direction.loc[resistance_mask] = -1

    score = pd.Series(0.0, index=ohlc.index)
    score.loc[support_mask] = (
        bars_above_streak.loc[support_mask]
        + bull_bounces.loc[support_mask] * BOUNCE_WEIGHT
    )
    score.loc[resistance_mask] = (
        bars_below_streak.loc[resistance_mask]
        + bear_rejects.loc[resistance_mask] * BOUNCE_WEIGHT
    )

    touch_count = pd.Series(0, index=ohlc.index, dtype="int64")
    touch_count.loc[support_mask] = bull_bounces.loc[support_mask].astype(int)
    touch_count.loc[resistance_mask] = bear_rejects.loc[resistance_mask].astype(int)

    bars_streak = pd.Series(0, index=ohlc.index, dtype="int64")
    bars_streak.loc[support_mask] = bars_above_streak.loc[support_mask]
    bars_streak.loc[resistance_mask] = bars_below_streak.loc[resistance_mask]

    return CandidateAnalysis(
        timeframe=timeframe,
        length=length,
        average_type=average_type,
        ohlc=ohlc,
        ma=ma,
        direction=direction,
        score=score,
        touch_count=touch_count,
        bars_streak=bars_streak,
    )


def resolve_position(length: int, as_of_pos: int) -> int | None:
    """Resolve a positive or negative series position."""
    position = length + as_of_pos if as_of_pos < 0 else as_of_pos
    if position < 0 or position >= length:
        return None
    return position


def finite_float(value: Any) -> float | None:
    """Return a regular float when the value is finite, otherwise None."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def snapshot_candidate(
    ticker: str,
    analysis: CandidateAnalysis,
    as_of_pos: int,
) -> DominantSnapshot | None:
    """Create a reportable snapshot when a candidate is dominant."""
    if len(analysis.ohlc) < 2:
        return None

    position = resolve_position(len(analysis.ohlc), as_of_pos)
    if position is None:
        return None

    direction_value = int(analysis.direction.iloc[position])
    if direction_value == 0:
        return None

    dominant_value = finite_float(analysis.ma.iloc[position])
    score = finite_float(analysis.score.iloc[position])
    if dominant_value is None or score is None or score <= 0:
        return None

    previous_close = finite_float(analysis.ohlc["Close"].iloc[-2])
    latest_close = finite_float(analysis.ohlc["Close"].iloc[-1])
    previous_ma = finite_float(analysis.ma.iloc[-2])
    latest_ma = finite_float(analysis.ma.iloc[-1])
    if None in {previous_close, latest_close, previous_ma, latest_ma}:
        return None

    return DominantSnapshot(
        ticker=ticker,
        timeframe=analysis.timeframe,
        length=analysis.length,
        average_type=analysis.average_type,
        direction="support" if direction_value == 1 else "resistance",
        score=score,
        touch_count=int(analysis.touch_count.iloc[position]),
        bars_streak=int(analysis.bars_streak.iloc[position]),
        latest_close=latest_close,
        previous_close=previous_close,
        dominant_value_latest=latest_ma,
        dominant_value_previous=previous_ma,
    )


def select_dominant_level(
    ticker: str,
    analyses: dict[tuple[str, int, str], CandidateAnalysis],
    as_of_pos: int,
) -> DominantSnapshot | None:
    """Select the dominant level by hierarchy, using score only within a step."""
    for timeframe, length in HIERARCHY:
        snapshots = [
            snapshot_candidate(ticker, analyses[(timeframe, length, average_type)], as_of_pos)
            for average_type in AVERAGE_TYPES
            if (timeframe, length, average_type) in analyses
        ]
        snapshots = [snapshot for snapshot in snapshots if snapshot is not None]
        if not snapshots:
            continue

        # Pine used ">" comparisons, so keep the first candidate on a score tie.
        return max(snapshots, key=lambda snapshot: snapshot.score)

    return None


def detect_break_signal(snapshot: DominantSnapshot) -> str | None:
    """Detect a true cross between the previous and latest completed candles."""
    if (
        snapshot.direction == "support"
        and snapshot.previous_close >= snapshot.dominant_value_previous
        and snapshot.latest_close < snapshot.dominant_value_latest
    ):
        return "break_below_dominant_support"

    if (
        snapshot.direction == "resistance"
        and snapshot.previous_close <= snapshot.dominant_value_previous
        and snapshot.latest_close > snapshot.dominant_value_latest
    ):
        return "break_above_dominant_resistance"

    return None


def passes_confirmation_filters(snapshot: DominantSnapshot) -> bool:
    """Extension point for future filters, such as RSI divergence confirmation."""
    return True


def clean_number(value: float | int | None, decimals: int = 4) -> float | int | None:
    """Convert numeric output to JSON/CSV friendly values."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, decimals)


def empty_result(
    ticker: str,
    status: str,
    message: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Build an empty result row with all output fields."""
    return {
        "ticker": ticker,
        "status": status,
        "signal_type": None,
        "dominant_timeframe": None,
        "dominant_length": None,
        "dominant_average_type": None,
        "dominant_direction": None,
        "latest_close": None,
        "previous_close": None,
        "dominant_value_latest": None,
        "dominant_value_previous": None,
        "score": None,
        "bounce_count": None,
        "bars_streak": None,
        "message": message,
        "error": error,
    }


def result_from_snapshot(
    snapshot: DominantSnapshot,
    status: str,
    signal_type: str | None,
    message: str,
) -> dict[str, Any]:
    """Build a result row from a dominant candidate snapshot."""
    return {
        "ticker": snapshot.ticker,
        "status": status,
        "signal_type": signal_type,
        "dominant_timeframe": snapshot.timeframe,
        "dominant_length": snapshot.length,
        "dominant_average_type": snapshot.average_type,
        "dominant_direction": snapshot.direction,
        "latest_close": clean_number(snapshot.latest_close),
        "previous_close": clean_number(snapshot.previous_close),
        "dominant_value_latest": clean_number(snapshot.dominant_value_latest),
        "dominant_value_previous": clean_number(snapshot.dominant_value_previous),
        "score": clean_number(snapshot.score, decimals=2),
        "bounce_count": snapshot.touch_count,
        "bars_streak": snapshot.bars_streak,
        "message": message,
        "error": None,
    }


def build_candidate_analyses(
    daily: pd.DataFrame,
) -> dict[tuple[str, int, str], CandidateAnalysis]:
    """Build all timeframe/length/SMA-EMA candidate analyses for one ticker."""
    analyses: dict[tuple[str, int, str], CandidateAnalysis] = {}

    for timeframe, rule in TIMEFRAME_RULES.items():
        ohlc = resample_ohlc(daily, rule)
        if len(ohlc) < 2:
            continue

        for length in {length for _, length in HIERARCHY}:
            for average_type in AVERAGE_TYPES:
                analyses[(timeframe, length, average_type)] = analyze_candidate(
                    ohlc=ohlc,
                    timeframe=timeframe,
                    length=length,
                    average_type=average_type,
                )

    return analyses


def scan_ticker(ticker: str) -> dict[str, Any]:
    """Scan one ticker and return one result row."""
    print(f"Scanning {ticker}...")

    try:
        daily = download_price_data(ticker)
        analyses = build_candidate_analyses(daily)
        if not analyses:
            return empty_result(
                ticker=ticker,
                status="no_signal",
                message="Not enough completed higher-timeframe data to analyze.",
            )

        # Use the previous completed candle as the active dominant level for
        # signal detection. Otherwise the break candle itself can reset the
        # dominance streak and hide the alert condition.
        previous_dominant = select_dominant_level(ticker, analyses, as_of_pos=-2)
        if previous_dominant is not None:
            signal_type = detect_break_signal(previous_dominant)
            if signal_type is not None and passes_confirmation_filters(previous_dominant):
                message = (
                    f"{ticker}: {human_signal_type(signal_type)} at "
                    f"{previous_dominant.timeframe} "
                    f"{previous_dominant.length}{previous_dominant.average_type}."
                )
                return result_from_snapshot(
                    previous_dominant,
                    status="signal",
                    signal_type=signal_type,
                    message=message,
                )

        current_dominant = select_dominant_level(ticker, analyses, as_of_pos=-1)
        if current_dominant is not None:
            message = (
                f"{ticker}: no break on dominant "
                f"{current_dominant.timeframe} "
                f"{current_dominant.length}{current_dominant.average_type} "
                f"{current_dominant.direction}."
            )
            return result_from_snapshot(
                current_dominant,
                status="no_signal",
                signal_type=None,
                message=message,
            )

        return empty_result(
            ticker=ticker,
            status="no_signal",
            message="No dominant support or resistance level found.",
        )

    except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the scan.
        return empty_result(
            ticker=ticker,
            status="error",
            message=f"{ticker}: scanner error.",
            error=str(exc),
        )


def write_results(results: list[dict[str, Any]], results_dir: Path = RESULTS_DIR) -> None:
    """Write scanner results to CSV and JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "latest_results.csv"
    json_path = results_dir / "latest_results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2)
        json_file.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def human_signal_type(signal_type: str | None) -> str:
    """Convert a signal identifier to readable text."""
    if signal_type == "break_below_dominant_support":
        return "Break below dominant support"
    if signal_type == "break_above_dominant_resistance":
        return "Break above dominant resistance"
    return "No signal"


def format_price(value: Any) -> str:
    """Format a value for console or Telegram output."""
    number = clean_number(value)
    return "n/a" if number is None else f"{number:g}"


def build_telegram_message(signals: list[dict[str, Any]]) -> str:
    """Build one grouped Telegram alert message."""
    sections = ["ALERT: Dominant MA/EMA Scanner Alerts"]

    for result in signals:
        count_label = (
            "Bounces"
            if result["dominant_direction"] == "support"
            else "Rejections"
        )
        sections.append(
            "\n".join(
                [
                    "",
                    str(result["ticker"]),
                    f"Signal: {human_signal_type(result['signal_type'])}",
                    "Dominant: "
                    f"{result['dominant_timeframe']} "
                    f"{result['dominant_length']}{result['dominant_average_type']}",
                    f"Close: {format_price(result['latest_close'])}",
                    f"Level: {format_price(result['dominant_value_latest'])}",
                    f"Score: {format_price(result['score'])}",
                    f"{count_label}: {result['bounce_count']}",
                    f"Bars streak: {result['bars_streak']}",
                ]
            )
        )

    return "\n".join(sections)


def send_telegram_message(message: str) -> bool:
    """Send a Telegram message when credentials are available."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print(
            "Warning: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing; "
            "skipping Telegram notification."
        )
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Warning: Telegram API request failed: {exc}")
        return False

    print("Telegram notification sent.")
    return True


def maybe_send_telegram(results: list[dict[str, Any]]) -> None:
    """Send alerts for signal results, or optionally send a no-signal message."""
    signals = [result for result in results if result["status"] == "signal"]
    if signals:
        send_telegram_message(build_telegram_message(signals))
        return

    if SEND_NO_SIGNAL_MESSAGE:
        send_telegram_message("Dominant MA/EMA Scanner: no signals found.")


def main() -> None:
    """Run the scanner."""
    print("Starting Dominant MA/EMA scanner.")
    print(f"Ticker file: {TICKERS_FILE}")
    print(f"Results directory: {RESULTS_DIR}")

    tickers = load_tickers()
    if not tickers:
        print("No tickers found.")
        write_results([])
        maybe_send_telegram([])
        return

    results = [scan_ticker(ticker) for ticker in tickers]
    write_results(results)
    maybe_send_telegram(results)

    signal_count = sum(1 for result in results if result["status"] == "signal")
    error_count = sum(1 for result in results if result["status"] == "error")
    print(
        f"Done. Tickers scanned: {len(results)}. "
        f"Signals: {signal_count}. Errors: {error_count}."
    )


if __name__ == "__main__":
    main()
