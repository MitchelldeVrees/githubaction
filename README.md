# Dominant MA/EMA Alerts

GitHub Actions based stock scanner:

```text
GitHub Actions
  -> Python scanner
  -> CSV / JSON results
  -> Telegram alert
```

The scanner downloads daily Yahoo Finance data, resamples it into higher-timeframe candles, checks dominant SMA/EMA support and resistance levels, writes `results/latest_results.csv` and `results/latest_results.json`, and sends Telegram alerts when a dominant level breaks or a completed candle closes near it.

This is not financial advice.

## Strategy

The strategy is translated from the supplied Pine Script:

- A bullish/support dominant level requires the close to stay above an MA/EMA for `BULL_BARS_REQUIRED` candles.
- A bearish/resistance dominant level requires the close to stay below an MA/EMA for `BEAR_BARS_REQUIRED` candles.
- Bounces/rejections are not required. They only increase the quality score.
- Support bounce: candle low is within `TOUCH_DISTANCE_PCT` above the MA/EMA and the candle closes above it.
- Resistance rejection: candle high is within `TOUCH_DISTANCE_PCT` below the MA/EMA and the candle closes below it.
- Support score: `bars_above_streak + bullish_bounces * BOUNCE_WEIGHT`.
- Resistance score: `bars_below_streak + bearish_rejections * BOUNCE_WEIGHT`.

The scanner checks:

- 20 SMA
- 50 SMA
- 20 EMA
- 50 EMA

Across:

- Weekly
- Biweekly
- Monthly
- Quarterly

Dominant level selection follows this hierarchy:

1. Weekly 20
2. Weekly 50
3. Biweekly 20
4. Biweekly 50
5. Monthly 20
6. Monthly 50
7. Quarterly 20
8. Quarterly 50

Within each hierarchy step, SMA and EMA are compared by score and the better one is selected. Score is not used to skip ahead in the hierarchy.

## Signal Logic

For support:

```text
previous_close >= previous_ma and latest_close < latest_ma
```

Signal type: `break_below_dominant_support`

For resistance:

```text
previous_close <= previous_ma and latest_close > latest_ma
```

Signal type: `break_above_dominant_resistance`

For proximity alerts, the latest completed candle must close on the valid side
of the dominant level and within `TOUCH_DISTANCE_PCT` of it:

```text
support: latest_close >= latest_ma and distance <= TOUCH_DISTANCE_PCT
resistance: latest_close <= latest_ma and distance <= TOUCH_DISTANCE_PCT
```

Signal types: `near_dominant_support` and `near_dominant_resistance`. A break
alert takes precedence over a proximity alert.

The scanner uses only completed higher-timeframe candles. If the current weekly, biweekly, monthly, or quarterly candle is still open, it is excluded.

## Data

The first version uses `yfinance` because it is free and simple to run in GitHub Actions. By default, `scanner.py` sets `YFINANCE_AUTO_ADJUST=true`, so Yahoo Finance returns adjusted OHLC values that account for splits and dividends. Set `YFINANCE_AUTO_ADJUST=false` if you want raw Yahoo OHLC data instead.

Free Yahoo Finance data can be delayed, revised, rate limited, or temporarily unavailable. International tickers should use Yahoo suffixes, such as `ASML.AS` or `WKL.AS`.

For larger ticker lists, the scanner downloads Yahoo data in batches, retries failed batches with backoff, waits briefly between batches, and stores daily OHLCV data in `.cache/yfinance`. GitHub Actions restores and saves that cache between runs, so after the first full-history scan, normal weekly runs only request recent incremental data for cached tickers.

## Edit Tickers

Edit `tickers.txt` and put one Yahoo Finance ticker per line:

```text
AAPL
MSFT
NVDA
ASML.AS
WKL.AS
```

Blank lines and lines starting with `#` are ignored.

## Run Locally

Install Python 3.11 or newer, then run:

```bash
pip install -r requirements.txt
python scanner.py
```

Results are written to:

- `results/latest_results.csv`
- `results/latest_results.json`

## Configuration

Main parameters are constants at the top of `scanner.py` and can also be overridden by environment variables:

```text
BULL_BARS_REQUIRED=70
BEAR_BARS_REQUIRED=40
TOUCH_DISTANCE_PCT=1.5
BOUNCE_LOOKBACK=150
BOUNCE_WEIGHT=10
SEND_NO_SIGNAL_MESSAGE=true
TICKERS_FILE=tickers.txt
SECURITY_IDENTIFIERS_FILE=security_identifiers.csv
RESULTS_DIR=results
YFINANCE_PERIOD=max
YFINANCE_AUTO_ADJUST=true
YFINANCE_INCREMENTAL_PERIOD=3mo
YFINANCE_BATCH_SIZE=50
YFINANCE_MAX_RETRIES=3
YFINANCE_RETRY_SLEEP_SECONDS=10
YFINANCE_BATCH_DELAY_SECONDS=2
YFINANCE_THREADS=false
ENABLE_PRICE_CACHE=true
DATA_CACHE_DIR=.cache/yfinance
CACHE_STALE_DAYS=45
TELEGRAM_MAX_MESSAGE_CHARS=3900
```

The code also has a `passes_confirmation_filters()` function where an optional RSI divergence filter can be added later without rewriting the scanner flow.

## Market and ISIN identifiers

Each result and Telegram alert includes `market` and `isin`. The scanner infers
the market from the Yahoo ticker suffix (for example, `.AS` is Euronext
Amsterdam and `.T` is Tokyo); symbols without a suffix are reported as a US
listing. Supply exact market names and ISINs in `security_identifiers.csv` to
override that fallback:

```csv
ticker,market,isin
ASML.AS,Euronext Amsterdam,NL0010273215
AAPL,NASDAQ,US0378331005
```

Yahoo Finance price downloads do not consistently provide ISINs, so ISINs must
be maintained in this mapping file. Rows may omit either value; the market
fallback is still used when `market` is blank.

For 300+ tickers, start with the defaults. If Yahoo starts rate-limiting, lower `YFINANCE_BATCH_SIZE` to `25` and raise `YFINANCE_BATCH_DELAY_SECONDS` to `5` or `10`. If a workflow has not run for more than `CACHE_STALE_DAYS`, cached tickers get a full-history refresh instead of an incremental refresh.

## Telegram Alerts

The scanner uses these environment variables:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

If they are missing, the script does not crash. It still writes CSV and JSON results and prints a warning when it would otherwise send a Telegram message.

If signals are found, one Telegram message is sent with all signals grouped together. If no signals are found, the scanner sends:

```text
DominantAM has been run but nothing was found
```

Set `SEND_NO_SIGNAL_MESSAGE=false` if you want to turn that no-signal message off.

To test Telegram only, go to `Actions -> Dominant MA/EMA Scanner -> Run workflow`, turn on `telegram_test`, and run it. This sends:

```text
DominantAM Telegram test message
```

If the token or chat ID is wrong, that test run fails and prints the Telegram API error in the workflow logs.

## Create Telegram Bot

1. Open Telegram.
2. Search for `BotFather`.
3. Create a new bot with `/newbot`.
4. Copy the bot token.
5. Send a message to your new bot.
6. Open this URL in a browser, replacing `<token>` with your bot token:

```text
https://api.telegram.org/bot<token>/getUpdates
```

7. Find the `chat.id` value in the response.
8. Add the bot token and chat ID as GitHub Actions secrets.

## Add GitHub Secrets

In your GitHub repository:

1. Go to `Settings`.
2. Go to `Secrets and variables`.
3. Go to `Actions`.
4. Click `New repository secret`.
5. Add `TELEGRAM_BOT_TOKEN`.
6. Add `TELEGRAM_CHAT_ID`.

Optional scanner settings can be added under `Settings -> Secrets and variables -> Actions -> Variables`.

## GitHub Actions

The workflow is at `.github/workflows/scanner.yml`.

It:

- Runs every Friday at 20:00 Europe/Amsterdam.
- Supports manual runs through `workflow_dispatch`.
- Uses Python 3.11.
- Installs `requirements.txt`.
- Runs `python scanner.py`.
- Uploads CSV and JSON results as artifacts.
- Restores and saves `.cache/yfinance` so large ticker lists do not need full-history downloads every week.
- Can optionally commit updated result files back to the repository.

The default schedule is:

```yaml
- cron: "0 18 * * 5"
- cron: "0 19 * * 5"
```

GitHub cron uses UTC and does not understand Amsterdam daylight saving time. The workflow has two Friday cron entries and a small `Europe/Amsterdam` gate, so only the run that lands at Friday 20:00 Amsterdam actually scans.

## Manual Workflow Test

1. Push this repository to GitHub.
2. Open the repository on GitHub.
3. Go to `Actions`.
4. Select `Dominant MA/EMA Scanner`.
5. Click `Run workflow`.
6. Leave `commit_results` off for a scanner test run, or turn it on if you want the workflow to commit updated result files.
7. To test only Telegram, turn on `telegram_test`; this skips the full scanner and sends a test message.
8. Open the completed run and download the `dominant-ma-alert-results` artifact.

## Output Fields

Each result contains:

- `ticker`
- `market`: configured market, or one inferred from the Yahoo ticker suffix
- `isin`: configured ISIN, if supplied
- `status`: `signal`, `no_signal`, or `error`
- `signal_type`
- `dominant_timeframe`
- `dominant_length`
- `dominant_average_type`
- `dominant_direction`
- `latest_close`
- `previous_close`
- `dominant_value_latest`
- `dominant_value_previous`
- `score`
- `bounce_count`
- `bars_streak`
- `message`
- `error`

For resistance setups, `bounce_count` contains the rejection count.
