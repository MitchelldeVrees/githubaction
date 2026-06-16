# Dominant MA/EMA Alerts

GitHub Actions based stock scanner:

```text
GitHub Actions
  -> Python scanner
  -> CSV / JSON results
  -> Telegram alert
```

The scanner downloads daily Yahoo Finance data, resamples it into higher-timeframe candles, checks dominant SMA/EMA support and resistance levels, writes `results/latest_results.csv` and `results/latest_results.json`, and sends Telegram alerts when a dominant level breaks.

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

The scanner uses only completed higher-timeframe candles. If the current weekly, biweekly, monthly, or quarterly candle is still open, it is excluded.

## Data

The first version uses `yfinance` because it is free and simple to run in GitHub Actions. By default, `scanner.py` sets `YFINANCE_AUTO_ADJUST=true`, so Yahoo Finance returns adjusted OHLC values that account for splits and dividends. Set `YFINANCE_AUTO_ADJUST=false` if you want raw Yahoo OHLC data instead.

Free Yahoo Finance data can be delayed, revised, rate limited, or temporarily unavailable. International tickers should use Yahoo suffixes, such as `ASML.AS` or `WKL.AS`.

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
RESULTS_DIR=results
YFINANCE_PERIOD=max
YFINANCE_AUTO_ADJUST=true
```

The code also has a `passes_confirmation_filters()` function where an optional RSI divergence filter can be added later without rewriting the scanner flow.

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
6. Leave `commit_results` off for a test run, or turn it on if you want the workflow to commit updated result files.
7. Open the completed run and download the `dominant-ma-alert-results` artifact.

## Output Fields

Each result contains:

- `ticker`
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
