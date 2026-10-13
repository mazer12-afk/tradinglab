# TradingLab

An autonomous systematic-trading laboratory. It **researches** a zoo of trading
strategies (classic trend/momentum rules and deep-learning models from the
Momentum Transformer / DeePM line of work), **selects** the ones that survive an
honest out-of-sample protocol, and **paper-trades** them on live crypto market
prices: unattended, crash-safe, with every decision and fill journaled.

```
            ┌────────────── research (monthly) ───────────────┐
 Binance ──►│ walk-forward OOS ─► PSR / DSR selection ─► holdout│──► models/selection.json
 archive /  └──────────────────────────────────────────────────┘          │
 exchange                                                                  ▼
 REST APIs ─────────────► live engine (every 5 min) ── orders ──► paper broker ──► state/lab.sqlite
                           fills · funding · hourly marks · drawdown guard        state/journal/*.csv
                                                                                    reports/live.html
```

<!-- RESULTS -->

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU torch (deep models)
pip install -e ".[dev]"

tradinglab fetch                 # download ~9 years of hourly data for 30 coins (~150 MB, cached)
tradinglab research              # tournament → models/selection.json (+ trained live models)
tradinglab cycle --report        # one live paper-trading cycle (idempotent)
tradinglab status                # accounts, P&L, exposure
tradinglab run                   # autonomous daemon: trade forever, retrain weekly, re-research monthly
pytest -q                        # test suite (no network needed)
```

`tradinglab research --no-deep` runs the classic strategies only (seconds). The full
run with deep models takes ~1.5 h on 4 CPU cores and caches every walk-forward fold,
so a re-run only recomputes what changed.

## What is in the lab

### Data (`tradinglab.data`)
* **Binance public archive** (`data.binance.vision`): every 1h kline since 2017 and
  USD-M perpetual funding rates, discovered through the S3 listing API, downloaded
  in parallel and cached as parquet (handles the 2025 switch to µs timestamps).
* **Real-time REST adapters**: Binance, Binance data-API, Binance.US, OKX, Coinbase
  and Kraken klines + best bid/ask, probed and used in order of availability.
* Universe: 30 liquid USDT pairs (BTC, ETH, SOL, XRP, …). New listings join the
  tradable cross-section only after 300 days of history.

### Strategies (`tradinglab.strategies`)
| name | idea |
|---|---|
| `hold_btc`, `basket_long` | benchmarks: vol-targeted BTC / equal-risk long basket (not eligible for selection) |
| `tsmom`, `tsmom_lf` | time-series momentum (Moskowitz–Ooi–Pedersen), 1/3/6/12-month blend; `_lf` = long/flat |
| `macd`, `macd_lf` | Baz et al. (2015) multi-scale MACD trend with the φ(x)=x·e^(−x²/4) response |
| `breakout`, `breakout_lf` | Donchian channel position, 20/55/100-day |
| `xsmom_63`, `xs_rev_5` | cross-sectional momentum / short-term reversal |
| `dl_lstm` | Deep Momentum Network (Lim, Zohren & Roberts 2019) |
| `dl_mom_trans` | **Momentum Transformer** (Wood et al. 2021): variable selection → LSTM → causal attention |
| `dl_deepm` | **DeePM-lite** (after [kieranjwood/deepm](https://github.com/kieranjwood/deepm), Wood et al. 2026): temporal backbone + cross-sectional attention, robust soft-min Sharpe objective |

Deep models are trained end-to-end to maximise the Sharpe ratio **net of
transaction costs** (turnover-penalised), with seed ensembles and early stopping
on a validation tail. DeePM's macro graph prior is omitted (no economic graph for
crypto); its lagged cross-section exists for asynchronous futures closes and is
unnecessary for 24/7 crypto bars that close simultaneously.

All strategies emit signals in [−1, 1]; one shared sizing layer turns them into
weights: per-asset volatility normalisation, portfolio volatility targeting (25%
annualised, EWMA covariance), caps of 35% per asset and 1.5× gross.

### Research protocol (`tradinglab.research`)
1. **Walk-forward**: deep models are retrained every 6 months on an expanding
   window; every signal from 2021-01 onwards is out-of-sample.
2. **Identical frictions** for everyone: 5 bps fee + 5 bps slippage per side, real
   perpetual funding rates, and a 3-hour signal→fill delay (daily bars close at
   00:00 UTC, trades fill at 03:00 UTC).
3. **Selection** on 2021-01 → 2024-12 only, requiring OOS Sharpe ≥ 0.5,
   Probabilistic Sharpe ≥ 0.8 and Deflated Sharpe ≥ 0.5, where the DSR deflates
   by the *effective* number of trials (correlation-adjusted); then a correlation
   filter and inverse-volatility weights.
4. **Holdout**: the frozen selection is evaluated on 2025-01 → today, data the
   selection never saw. That is the honest estimate of live performance.
5. For live trading, the same rules are re-applied to the full OOS history.

### Live engine (`tradinglab.live`)
* `tradinglab cycle` is **idempotent**: run it from cron, a daemon or CI at any
  frequency. Each cycle replays every event that became knowable since the last
  one, in time order (fills, 8-hourly funding, hourly marks), then makes new
  decisions when a new daily bar is complete.
* **Feed modes** (`live.feed: auto`):
  * *realtime*: exchange APIs reachable; history is topped up with the latest
    hourly klines and orders fill immediately at the live bid/ask.
  * *archive*: only the public archive is reachable (e.g. a sandbox with strict
    egress). Decisions are still taken in real time; orders are timestamped *now*
    and filled at the first 1-minute trade price after submission once the
    exchange publishes it (~3 h after each UTC midnight). It is a genuine forward
    test on live prices, revealed with a delay and never back-filled with
    hindsight.
* **Accounts**: `main` trades the selected ensemble; `shadow:<strategy>` accounts
  paper-trade every candidate, which makes it a live tournament.
* **Risk**: drawdown guard (de-risk from −15%, flatten at −30%, 20-bar cool-down,
  then resume), stale-data refusal (> 60 h), bad-tick filter, per-asset and gross
  caps, minimum order size, and an operator kill switch (`touch state/HALT`).
* **Persistence**: one SQLite file (`state/lab.sqlite`) with balances, positions,
  orders, fills, funding, equity marks, decisions and an event journal; CSV
  exports in `state/journal/` and `state/status.json`. Restarting from the file
  reproduces an uninterrupted run exactly (tested).
* **Autonomy**: `tradinglab run` loops forever, survives any exception with
  back-off, writes `state/heartbeat.json` for watchdogs, retrains the live deep
  models weekly and re-runs the full research monthly in a subprocess, then
  hot-reloads the new selection.

### Tests
`pytest -q` (no network) covers, among other things:
* **no look-ahead**: features, every classic strategy, sizing and the deep models
  are unchanged when future data is appended;
* **backtest ↔ live parity**: the live engine fed the same data reproduces the
  backtester's equity to within 0.02% (perp, spot, with funding);
* deferred archive-mode fills at the correct post-decision price;
* crash/restart recovery; broker accounting; drawdown guard; kill switch;
  payload parsers for the archive and all exchanges.

`RUN_NETWORK_TESTS=1 pytest tests/test_network.py` checks live endpoints from
the machine that will trade.

## Deployment

* **systemd**: `deploy/tradinglab.service`.
* **Docker**: `docker compose -f deploy/docker-compose.yml up -d` (CPU torch,
  healthcheck on the heartbeat file; mount `state/`, `data/`, `models/`).
* **GitHub Actions** (`.github/workflows/paper-trade.yml`): hourly `tradinglab
  cycle` on GitHub runners (realtime mode) that commits `state/` back to the repo.
  It is opt-in, because scheduled workflows run only from the default branch and
  use Actions minutes: set the repository variable `PAPER_TRADING_ENABLED=true`.

Configuration lives in `config/default.yaml`; pass `--config my.yaml` to layer
overrides. `market.instrument: spot` switches to long-only spot accounting.

## Caveats

* **Paper trading only.** No order is ever sent to an exchange.
* Short live windows say almost nothing about an edge: daily P&L is dominated by
  market moves. Judge strategies on the out-of-sample and holdout statistics,
  and on months of live data.
* The universe is today's surviving listings, so backtests carry survivorship
  bias (coins like LUNA or FTT are absent).
* Funding for the current month is estimated (trailing mean) until Binance
  archives the month; estimated payments are flagged in the journal.
* Perpetual prices are proxied by spot prices (basis is typically a few bps).
