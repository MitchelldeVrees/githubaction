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

The dominance rules follow the JDB strategy (component 1, the MA component).

An average is **dominant support** when all of these hold:

- the close has stayed above it for at least `BULL_BARS_REQUIRED` candles;
- price has touched it at least `MIN_TOUCHES` times during that streak. A touch
  is a candle whose low comes within `TOUCH_DISTANCE_PCT` of the average and
  which still closes above it. A long streak far above an average is not
  support;
- at most `FAULTY_CANDLES_ALLOWED` (0 or 1) earlier candle closed below it, and
  only as a close call, within `FAULTY_CANDLE_MAX_PCT` of the average. Such a
  candle is treated as noise and does not end the streak. A larger miss, or a
  second one, does.

**Dominant resistance** is the mirror image with `BEAR_BARS_REQUIRED` candles
below the average. Candles before an average has a full window behind it never
count.

The scanner checks SMA and EMA of the lengths in `DOMINANT_LENGTHS` (20, 50 and
200) on weekly, biweekly, monthly and quarterly candles. Selection walks the
timeframes from short to long and, within a timeframe, the lengths from short to
long: Weekly 20, Weekly 50, Weekly 200, Biweekly 20, and so on. The first step
with a dominant level decides. Within a step, SMA and EMA are compared by score
(`streak + touches in the last BOUNCE_LOOKBACK candles * BOUNCE_WEIGHT`).

The numbers behind "several touches" and "a close call" are this project's
reading of rules JDB states in words:

```text
BULL_BARS_REQUIRED=70
BEAR_BARS_REQUIRED=40
MIN_TOUCHES=2
FAULTY_CANDLES_ALLOWED=1
FAULTY_CANDLE_MAX_PCT=2.0
DOMINANT_LENGTHS=20,50,200
```

To get the original behavior back, where an uninterrupted streak alone made an
average dominant, set `MIN_TOUCHES=0`, `FAULTY_CANDLES_ALLOWED=0` and
`DOMINANT_LENGTHS=20,50`.

Each result also reports `streak_touches` and `faulty_candles` for the dominant
level.

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

## Alert Context

Every result row and Telegram alert also carries context about the chart. These
fields are informational and never change which signals are raised:

- `candle_date`: end date of the completed candle the result is based on. A
  biweekly, monthly, or quarterly alert can refer to a candle that closed before
  the latest Friday.
- `ma200_value`, `ma200_distance_pct`: the 200-period average on the dominant
  timeframe, in the dominant level's SMA/EMA type, and its distance from the
  latest close.
- `next_target`, `next_target_value`, `next_target_distance_pct`: for a break
  only, the next longer average (20 -> 50 -> 200) that still lies in the
  direction of the break. Averages the close has already passed are skipped.

- `rsi`, `negative_divergence`, `positive_divergence`: RSI on the dominant
  timeframe and whether it diverges from price. Negative divergence: the latest
  swing high is higher than the previous one while RSI is lower. Positive
  divergence: the latest swing low is lower while RSI is higher. A Telegram
  alert shows the one that matters for its level: negative under a support
  level, positive above a resistance level.

An average is reported only when the timeframe has at least that many completed
candles, so the 200-period average is empty for short histories.

Divergence is an approximation of what a trader draws by eye. A swing is a
candle whose high or low is the extreme of `DIVERGENCE_SWING_BARS` candles on
each side; the last two swings within `DIVERGENCE_LOOKBACK` candles are
compared; the latest must be at most `DIVERGENCE_MAX_AGE` candles old.

```text
RSI_LENGTH=14
DIVERGENCE_SWING_BARS=5
DIVERGENCE_LOOKBACK=100
DIVERGENCE_MAX_AGE=26
```

## Data

The first version uses `yfinance` because it is free and simple to run in GitHub Actions. By default, `scanner.py` sets `YFINANCE_AUTO_ADJUST=true`, so Yahoo Finance returns adjusted OHLC values that account for splits and dividends. Set `YFINANCE_AUTO_ADJUST=false` if you want raw Yahoo OHLC data instead.

Free Yahoo Finance data can be delayed, revised, rate limited, or temporarily unavailable. International tickers should use Yahoo suffixes, such as `ASML.AS` or `WKL.AS`.

For larger ticker lists, the scanner downloads Yahoo data in batches, retries failed batches with backoff, waits briefly between batches, and stores daily OHLCV data in `.cache/yfinance`. GitHub Actions restores and saves that cache between runs, so after the first full-history scan, normal weekly runs only request recent incremental data for cached tickers.

## Backtest

`backtest.py` replays the scanner over the full price history and simulates one
trade per break it would have alerted on. It imports the scanner's own candle
building and dominance analysis, so it tests the same signals the Friday run
raises.

```bash
python backtest.py                      # stop beyond the break candle
python backtest.py --stop reclaim       # stop on a close back across the average
python backtest.py --stop trail         # candle stop that trails the dominant average
python backtest.py --start 2010-01-01   # ignore older signals
YFINANCE_AUTO_ADJUST=false python backtest.py   # unadjusted prices
```

Rules: entry at the close of the break candle (short below support, long above
resistance); target is the next longer average (20 -> 50 -> 200) on the same
timeframe that lies in the break direction; a trade open after `--max-candles`
candles (default 104) exits at the close. Costs default to 0.1% per side.

`--stop trail` starts with the candle stop. Once a candle has closed at least
`--trail-after-r` (default 1) times the initial risk in profit, the stop moves
to just beyond the dominant average and follows it, only ever tightening. Exits
on the moved stop are counted as `trail` in the summary.

Two entries are simulated, chosen with `--setups` (default `break,pullback`):

- `break`: the close through the dominant average described above.
- `pullback`: while an average is dominant, price coming back to it. For
  dominant support this is a long: entry `--pullback-entry-atr` (default 0.25)
  ATR above the average, stop `--pullback-stop-atr` (default 0.75) ATR below
  it. ATR is the 14-candle average true range of the signal's own timeframe, so
  a monthly stop sits further from the average than a weekly one and a volatile
  stock gets more room than a quiet one. The stop follows the average and only
  tightens. The target is the nearest other average above the entry; when price
  is above all of them there is no target and the trailed stop ends the trade.
  Dominant resistance is the mirror image, a short. One position per idea at a
  time, where the SMA and EMA of the same timeframe and length are one idea;
  after three losing attempts in a row on an idea, no new attempt for 26
  candles.

The entry and stop of a pullback use the average of the last completed candle,
and the trade is entered inside the next candle, which is how a touch of the
average is traded live. A candle that falls through both entry and stop counts
as a full loss.

The summary reports the two entries separately (`break short`, `pullback long`,
and so on).

Each trade records RSI and divergence on the break candle. Divergence filters
nothing; the summary has `short with divergence`, `short without divergence`,
`long with divergence` and `long without divergence` rows, where "with" means
negative divergence for a short and positive divergence for a long.

Output goes to `results/backtest/`: `backtest_trades.csv` with every trade and
`backtest_summary.json` with win rate, average win and loss, expectancy, average
R, holding time and the longest losing streak, overall and per dominant level.

Read the results with these limits in mind:

- `tickers.txt` holds companies that exist today. Companies that failed are
  missing, which flatters long trades and penalizes short trades.
- Monthly and quarterly levels produce few trades per ticker.
- Entries assume a fill at the closing price of the break candle.
- Retracement entries, Bollinger entries and options are not tested.
- "A little above" and "a little below" the average are not quantified by JDB;
  the 0.25 and 0.75 ATR defaults are this project's choice.

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
- cron: "0 20 * * 5"
  timezone: "Europe/Amsterdam"
```

The timezone-aware schedule handles Amsterdam daylight saving time directly, so no additional runtime clock check is needed.

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
