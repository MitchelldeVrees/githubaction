"""Write the largest tickers of a ticker file, by today's market cap in US dollars.

Used by the backtest workflow to compare the full ticker list with only the
largest companies. Picking by today's size favors companies that grew, so a
backtest on this list is expected to look better than reality (survivorship
bias); it is meant to be compared with the full list, not to replace it.

    python top_tickers.py --count 100 --out tickers_top100.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yfinance as yf

import scanner

# Quote currencies that Yahoo gives in a minor unit.
MINOR_UNITS = {"GBp": ("GBP", 100.0), "GBX": ("GBP", 100.0), "ILA": ("ILS", 100.0), "ZAc": ("ZAR", 100.0)}


def usd_rate(currency: str, cache: dict[str, float]) -> float | None:
    """US dollars per one unit of `currency`."""
    if currency == "USD":
        return 1.0
    if currency not in cache:
        try:
            cache[currency] = float(yf.Ticker(f"{currency}USD=X").fast_info["last_price"])
        except Exception:  # noqa: BLE001 - an unknown currency only drops its tickers.
            cache[currency] = float("nan")
    rate = cache[currency]
    return rate if rate == rate else None


def market_cap_usd(ticker: str, rates: dict[str, float]) -> float | None:
    """Today's market cap of `ticker` in US dollars, or None if Yahoo has none."""
    try:
        info = yf.Ticker(ticker).fast_info
        cap = info["market_cap"]
        currency = info["currency"]
    except Exception:  # noqa: BLE001 - a missing ticker is skipped.
        return None
    if not cap or not currency:
        return None
    currency, divisor = MINOR_UNITS.get(currency, (currency, 1.0))
    rate = usd_rate(currency, rates)
    return None if rate is None else float(cap) / divisor * rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tickers-file", type=Path, default=scanner.TICKERS_FILE)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("tickers_top100.txt"))
    args = parser.parse_args()

    rates: dict[str, float] = {}
    caps: dict[str, float] = {}
    for ticker in scanner.load_tickers(args.tickers_file):
        cap = market_cap_usd(ticker, rates)
        if cap is not None:
            caps[ticker] = cap
        time.sleep(0.2)

    ranked = sorted(caps, key=caps.get, reverse=True)[: args.count]
    lines = [
        f"# The {len(ranked)} largest of {args.tickers_file.name} by market cap in USD,",
        f"# ranked on {time.strftime('%Y-%m-%d')}, largest first. Caps in the .csv beside it.",
    ]
    lines += ranked
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", encoding="utf-8") as handle:
        handle.write("rank,ticker,market_cap_usd_bn\n")
        for rank, ticker in enumerate(ranked, 1):
            handle.write(f"{rank},{ticker},{caps[ticker] / 1e9:.1f}\n")
    print("\n".join(lines))
    print(f"{len(caps)} of the tickers had a market cap.")


if __name__ == "__main__":
    main()
