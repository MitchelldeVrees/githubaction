"""Dominant SMA/EMA stock scanner.

This script translates the supplied TradingView Pine Script dominance logic
into a GitHub Actions friendly Python scanner. It downloads daily Yahoo Finance
data, resamples it into higher-timeframe candles, selects the dominant level by
the requested hierarchy, writes CSV/JSON results, and optionally sends Telegram
alerts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
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

SEND_NO_SIGNAL_MESSAGE = env_bool("SEND_NO_SIGNAL_MESSAGE", True)
TICKERS_FILE = env_path("TICKERS_FILE", "tickers.txt")
RESULTS_DIR = env_path("RESULTS_DIR", "results")

# yfinance with auto_adjust=True returns split/dividend adjusted OHLC values.
YFINANCE_PERIOD = env_str("YFINANCE_PERIOD", "max")
YFINANCE_AUTO_ADJUST = env_bool("YFINANCE_AUTO_ADJUST", True)
YFINANCE_INCREMENTAL_PERIOD = env_str("YFINANCE_INCREMENTAL_PERIOD", "3mo")
YFINANCE_BATCH_SIZE = env_int("YFINANCE_BATCH_SIZE", 50)
YFINANCE_MAX_RETRIES = env_int("YFINANCE_MAX_RETRIES", 3)
YFINANCE_RETRY_SLEEP_SECONDS = env_float("YFINANCE_RETRY_SLEEP_SECONDS", 10.0)
YFINANCE_BATCH_DELAY_SECONDS = env_float("YFINANCE_BATCH_DELAY_SECONDS", 2.0)
YFINANCE_THREADS = env_bool("YFINANCE_THREADS", False)

ENABLE_PRICE_CACHE = env_bool("ENABLE_PRICE_CACHE", True)
DATA_CACHE_DIR = env_path("DATA_CACHE_DIR", ".cache/yfinance")
CACHE_STALE_DAYS = env_int("CACHE_STALE_DAYS", 45)

TELEGRAM_MAX_MESSAGE_CHARS = env_int("TELEGRAM_MAX_MESSAGE_CHARS", 3900)


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


def chunked(items: list[str], size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks."""
    chunk_size = max(1, size)
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def clean_price_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV columns, index, ordering, and numeric values."""
    if data.empty:
        return data

    data = data.rename(columns={column: str(column).title() for column in data.columns})
    missing = [column for column in OHLC_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {', '.join(missing)}")

    data = data[OHLC_COLUMNS].copy()
    for column in OHLC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data.sort_index()
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_convert(None)
    data.index.name = "Date"
    return data


def normalize_yfinance_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a single-ticker OHLCV DataFrame from yfinance output."""
    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        for level in range(data.columns.nlevels):
            if ticker in data.columns.get_level_values(level):
                data = data.xs(ticker, axis=1, level=level)
                break

    while isinstance(data.columns, pd.MultiIndex) and data.columns.nlevels > 1:
        dropped_level = False
        for level in range(data.columns.nlevels):
            if len(set(data.columns.get_level_values(level))) == 1:
                data = data.copy()
                data.columns = data.columns.droplevel(level)
                dropped_level = True
                break
        if not dropped_level:
            data = data.copy()
            data.columns = data.columns.get_level_values(-1)
            break

    return clean_price_frame(data)


def cache_mode_dir() -> Path:
    """Return the cache directory for the current OHLC adjustment mode."""
    mode = "adjusted" if YFINANCE_AUTO_ADJUST else "raw"
    return DATA_CACHE_DIR / mode


def cache_file_for_ticker(ticker: str) -> Path:
    """Return a deterministic cache path for a Yahoo ticker."""
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in ticker.upper()
    ).strip("_")
    safe_name = safe_name or "ticker"
    digest = hashlib.sha1(ticker.upper().encode("utf-8")).hexdigest()[:10]
    return cache_mode_dir() / f"{safe_name}_{digest}.csv"


def read_cached_price_data(ticker: str) -> pd.DataFrame | None:
    """Read cached daily OHLCV data for one ticker."""
    if not ENABLE_PRICE_CACHE:
        return None

    cache_path = cache_file_for_ticker(ticker)
    if not cache_path.exists():
        return None

    try:
        data = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        data = clean_price_frame(data)
    except Exception as exc:  # noqa: BLE001 - corrupt cache should not stop a run.
        print(f"Warning: ignoring unreadable cache for {ticker}: {exc}")
        return None

    return data if not data.empty else None


def write_cached_price_data(ticker: str, data: pd.DataFrame) -> None:
    """Write daily OHLCV data to the local ticker cache."""
    if not ENABLE_PRICE_CACHE or data.empty:
        return

    cache_path = cache_file_for_ticker(ticker)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = clean_price_frame(data)
    cached.to_csv(cache_path, index_label="Date")


def latest_data_age_days(data: pd.DataFrame) -> int | None:
    """Return the age in days of the latest cached candle."""
    if data.empty:
        return None

    latest_date = pd.to_datetime(data.index.max()).tz_localize(None).normalize()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    return int((today - latest_date).days)


def cache_needs_full_refresh(data: pd.DataFrame | None) -> bool:
    """Decide whether cached data is too old for an incremental update."""
    if data is None or data.empty:
        return True

    age_days = latest_data_age_days(data)
    return age_days is None or age_days > CACHE_STALE_DAYS


def merge_price_data(
    cached: pd.DataFrame | None,
    fresh: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cached and fresh OHLCV data, preferring fresh duplicate rows."""
    if cached is None or cached.empty:
        return clean_price_frame(fresh)

    combined = pd.concat([cached, fresh])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return clean_price_frame(combined)


def extract_ticker_from_download(
    downloaded: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Extract one ticker's OHLCV frame from a yfinance batch response."""
    if downloaded.empty:
        return downloaded
    return normalize_yfinance_columns(downloaded.copy(), ticker)


def download_yfinance_once(tickers: list[str], period: str) -> pd.DataFrame:
    """Call yfinance once for a ticker batch."""
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=YFINANCE_AUTO_ADJUST,
        progress=False,
        group_by="ticker",
        threads=YFINANCE_THREADS,
    )
    return data


def sleep_before_retry(attempt: int) -> None:
    """Sleep with exponential backoff between Yahoo retries."""
    sleep_seconds = YFINANCE_RETRY_SLEEP_SECONDS * (2 ** (attempt - 1))
    if sleep_seconds <= 0:
        return
    print(f"Waiting {sleep_seconds:g}s before retry...")
    time.sleep(sleep_seconds)


def download_price_batches(
    tickers: list[str],
    period: str,
    batch_label: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Download ticker data in batches with retries and request spacing."""
    data_by_ticker: dict[str, pd.DataFrame] = {}
    errors_by_ticker: dict[str, str] = {}
    if not tickers:
        return data_by_ticker, errors_by_ticker

    batches = chunked(tickers, YFINANCE_BATCH_SIZE)
    max_retries = max(1, YFINANCE_MAX_RETRIES)

    for batch_index, batch in enumerate(batches, start=1):
        remaining = list(batch)
        last_error = "Yahoo Finance returned no OHLC data"
        print(
            f"Downloading {batch_label} batch {batch_index}/{len(batches)} "
            f"({len(batch)} tickers, period={period})..."
        )

        for attempt in range(1, max_retries + 1):
            try:
                downloaded = download_yfinance_once(remaining, period)
                recovered: list[str] = []

                for ticker in remaining:
                    try:
                        ticker_data = extract_ticker_from_download(downloaded, ticker)
                    except Exception as exc:  # noqa: BLE001 - retry ticker below.
                        errors_by_ticker[ticker] = str(exc)
                        continue

                    if ticker_data.empty:
                        errors_by_ticker[ticker] = "Yahoo Finance returned no OHLC data"
                        continue

                    data_by_ticker[ticker] = ticker_data
                    recovered.append(ticker)
                    errors_by_ticker.pop(ticker, None)

                remaining = [ticker for ticker in remaining if ticker not in recovered]
                if not remaining:
                    break

                last_error = (
                    f"Yahoo Finance returned no OHLC data for "
                    f"{len(remaining)} ticker(s)"
                )

            except Exception as exc:  # noqa: BLE001 - retry batch below.
                last_error = str(exc)

            if remaining and attempt < max_retries:
                print(
                    f"Retry {attempt}/{max_retries - 1} for "
                    f"{len(remaining)} ticker(s). Last error: {last_error}"
                )
                sleep_before_retry(attempt)

        for ticker in remaining:
            errors_by_ticker[ticker] = errors_by_ticker.get(ticker, last_error)

        if batch_index < len(batches) and YFINANCE_BATCH_DELAY_SECONDS > 0:
            time.sleep(YFINANCE_BATCH_DELAY_SECONDS)

    return data_by_ticker, errors_by_ticker


def download_price_data_bulk(
    tickers: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Download and cache daily OHLCV data for all tickers."""
    unique_tickers = list(dict.fromkeys(tickers))
    cached_by_ticker: dict[str, pd.DataFrame] = {}
    full_history_tickers: list[str] = []
    incremental_tickers: list[str] = []

    for ticker in unique_tickers:
        cached = read_cached_price_data(ticker)
        if cached is not None:
            cached_by_ticker[ticker] = cached

        if ENABLE_PRICE_CACHE and cached is not None and not cache_needs_full_refresh(cached):
            incremental_tickers.append(ticker)
        else:
            full_history_tickers.append(ticker)

    print(
        "Price cache: "
        f"{len(incremental_tickers)} ticker(s) will use incremental refresh, "
        f"{len(full_history_tickers)} ticker(s) need full history."
    )

    full_data, full_errors = download_price_batches(
        full_history_tickers,
        YFINANCE_PERIOD,
        "full-history",
    )
    incremental_data, incremental_errors = download_price_batches(
        incremental_tickers,
        YFINANCE_INCREMENTAL_PERIOD,
        "incremental",
    )

    price_data_by_ticker: dict[str, pd.DataFrame] = {}
    errors_by_ticker: dict[str, str] = {}

    for ticker in unique_tickers:
        cached = cached_by_ticker.get(ticker)
        if ticker in full_history_tickers:
            fresh = full_data.get(ticker)
            source_errors = full_errors
        else:
            fresh = incremental_data.get(ticker)
            source_errors = incremental_errors

        if fresh is not None and not fresh.empty:
            merged = merge_price_data(cached, fresh)
            if not merged.empty:
                price_data_by_ticker[ticker] = merged
                write_cached_price_data(ticker, merged)
                continue

        errors_by_ticker[ticker] = source_errors.get(
            ticker,
            "Yahoo Finance returned no OHLC data",
        )

    return price_data_by_ticker, errors_by_ticker


def download_price_data(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV data for one Yahoo Finance ticker."""
    data_by_ticker, errors_by_ticker = download_price_data_bulk([ticker])
    if ticker in data_by_ticker:
        return data_by_ticker[ticker]
    raise ValueError(errors_by_ticker.get(ticker, "Yahoo Finance returned no OHLC data"))


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


def scan_ticker(
    ticker: str,
    daily: pd.DataFrame | None = None,
    data_error: str | None = None,
) -> dict[str, Any]:
    """Scan one ticker and return one result row."""
    print(f"Scanning {ticker}...")

    try:
        if data_error is not None:
            return empty_result(
                ticker=ticker,
                status="error",
                message=f"{ticker}: price data error.",
                error=data_error,
            )

        if daily is None:
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


def build_signal_telegram_section(result: dict[str, Any]) -> str:
    """Build one ticker section for a Telegram alert."""
    count_label = (
        "Bounces"
        if result["dominant_direction"] == "support"
        else "Rejections"
    )
    return "\n".join(
        [
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


def build_telegram_messages(signals: list[dict[str, Any]]) -> list[str]:
    """Build grouped Telegram alert messages under Telegram's text limit."""
    header = "ALERT: Dominant MA/EMA Scanner Alerts"
    messages: list[str] = []
    current_message = header

    for result in signals:
        section = build_signal_telegram_section(result)
        candidate_message = f"{current_message}\n\n{section}"
        if (
            len(candidate_message) > TELEGRAM_MAX_MESSAGE_CHARS
            and current_message != header
        ):
            messages.append(current_message)
            current_message = f"{header}\n\n{section}"
        else:
            current_message = candidate_message

    messages.append(current_message)
    return messages


def build_telegram_message(signals: list[dict[str, Any]]) -> str:
    """Build one grouped Telegram alert message for backward compatibility."""
    return build_telegram_messages(signals)[0]


def send_telegram_message(message: str, fail_on_error: bool = False) -> bool:
    """Send a Telegram message when credentials are available."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        warning = (
            "Warning: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing; "
            "skipping Telegram notification."
        )
        print(warning)
        if fail_on_error:
            raise RuntimeError(warning)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message},
            timeout=20,
        )
        if not response.ok:
            warning = (
                "Telegram API request failed with "
                f"HTTP {response.status_code}: {response.text[:500]}"
            )
            print(f"Warning: {warning}")
            if fail_on_error:
                raise RuntimeError(warning)
            return False
    except requests.RequestException as exc:
        error_text = str(exc)
        if token:
            error_text = error_text.replace(token, "<redacted>")
        warning = f"Telegram API request failed: {type(exc).__name__}: {error_text}"
        print(f"Warning: {warning}")
        if fail_on_error:
            raise RuntimeError(warning) from exc
        return False

    print("Telegram notification sent.")
    return True


def run_telegram_test() -> None:
    """Send a Telegram test message and fail with diagnostics on error."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print(f"TELEGRAM_BOT_TOKEN configured: {'yes' if token else 'no'}")
    print(f"TELEGRAM_CHAT_ID configured: {'yes' if chat_id else 'no'}")

    try:
        send_telegram_message("DominantAM Telegram test message", fail_on_error=True)
    except RuntimeError as exc:
        print(f"Telegram test failed: {exc}")
        raise SystemExit(1) from exc

    print("Telegram test succeeded.")


def maybe_send_telegram(results: list[dict[str, Any]]) -> None:
    """Send alerts for signal results, or optionally send a no-signal message."""
    signals = [result for result in results if result["status"] == "signal"]
    if signals:
        for message in build_telegram_messages(signals):
            send_telegram_message(message)
        return

    if SEND_NO_SIGNAL_MESSAGE:
        send_telegram_message("DominantAM has been run but nothing was found")


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

    price_data_by_ticker, data_errors_by_ticker = download_price_data_bulk(tickers)
    results = [
        scan_ticker(
            ticker,
            daily=price_data_by_ticker.get(ticker),
            data_error=data_errors_by_ticker.get(ticker),
        )
        for ticker in tickers
    ]
    write_results(results)
    maybe_send_telegram(results)

    signal_count = sum(1 for result in results if result["status"] == "signal")
    error_count = sum(1 for result in results if result["status"] == "error")
    print(
        f"Done. Tickers scanned: {len(results)}. "
        f"Signals: {signal_count}. Errors: {error_count}."
    )


if __name__ == "__main__":
    if "--telegram-test" in sys.argv:
        run_telegram_test()
    else:
        main()
