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

## Research results (data through 2026-10-12)

Walk-forward out-of-sample, **net of 10 bps per side, real funding and a 3-hour
signal→fill delay**, 30 coins, daily bars. Full OOS = 2021-01 → 2026-10;
selection window = 2021–2024; holdout = 2025-01 → 2026-10.

| strategy | OOS Sharpe | CAGR | max DD | turnover/yr | Sharpe 2021–24 | Sharpe 2025–26 |
|---|---|---|---|---|---|---|
| `tsmom_lf` | 0.86 | 21.4% | -35% | 16× | 1.25 | -0.06 |
| `breakout_lf` | 0.84 | 21.6% | -38% | 19× | 1.25 | -0.10 |
| `macd_lf` | 0.71 | 16.3% | -35% | 8× | 1.13 | -0.33 |
| `basket_long` (benchmark) | 0.69 | 15.0% | -36% | 2× | 0.97 | 0.03 |
| `tsmom` | 0.63 | 13.9% | -35% | 29× | 0.74 | 0.41 |
| `macd` | 0.61 | 13.1% | -33% | 14× | 0.78 | 0.24 |
| `breakout` | 0.57 | 11.9% | -33% | 37× | 0.89 | -0.21 |
| `hold_btc` (benchmark) | 0.41 | 6.1% | -38% | 2× | 0.56 | -0.04 |
| `dl_deepm` | 0.40 | 7.3% | -40% | 55× | 0.62 | -0.11 |
| `dl_lstm` | 0.20 | 1.8% | -42% | 40× | 0.26 | 0.06 |
| `xsmom_63` | -0.14 | -5.4% | -36% | 43× | 0.05 | -0.61 |
| `dl_mom_trans` | -0.37 | -14.3% | -67% | 117× | -0.29 | -0.60 |
| `xs_rev_5` | -1.44 | -29.5% | -91% | 175× | -1.36 | -1.64 |

**What is actually profitable.** Time-series trend following is the only family
with robust out-of-sample profits after costs. It comes in two nearly
uncorrelated flavours: long/flat trend (ρ≈0.9 with simply holding the basket,
i.e. timed beta) and long/short trend (ρ≈−0.1 with the market).
Cross-sectional momentum and short-term reversal lose money after costs.

**Deep models.** The Momentum Transformer, LSTM-DMN and DeePM-lite did *not* beat
the simple rules on ~5 years of crypto data (they were designed on 50 futures
× 30 years). Their positions were too jumpy, with turnover up to 117×/yr eating the
edge. They keep running as live shadow accounts, retrained weekly, so the lab
keeps measuring them.

**Selection → live portfolio.** Strategies are clustered by return correlation.
A cluster is chosen when its equal-weight stream clears Sharpe ≥ 0.5 and a
Deflated Sharpe ≥ 0.5 (trials = number of clusters). All eligible members are
held, equal-weighted inside the cluster, and clusters are split by inverse
volatility. The live `main` account holds two clusters, long/flat trend
(`tsmom_lf`, `macd_lf`, `breakout_lf`, 16% each) and long/short trend
(`tsmom`, `macd`, 26% each).
* Applying these rules over the full OOS period gives Sharpe 0.73, CAGR 16.3%,
  max drawdown -29%, +139% total.
* Selection frozen on 2021–24 and evaluated on 2025-01 → 2026-10 gives
  Sharpe 0.16, +1.7%, max drawdown -21%. The last 21 months were a poor regime
  for crypto trend following.

**Research log (honest disclosure).** A first selection rule (single-strategy
DSR with the cross-sectional variance, plus a 0.9 correlation filter) selected
long/flat trend only. That choice then lost 23.6% in the holdout, partly because
a −15%/−30% drawdown guard de-risked at the lows and missed the recoveries (it
also lowered Sharpe in the 2021–24 window). The guard is now a catastrophe
breaker (−25%/−40%) and selection is cluster-based. Both changes were made
*after* seeing that holdout, so the holdout above is no longer an untouched test.
Turnover controls (EMA-5 signal smoothing, 0.5% rebalance band) were chosen on
2021–24 data only.

Full tables: `reports/research/` (`index.html`, `candidates_oos.csv`,
`selection_window.csv`, `holdout_window.csv`, `equity_curves.csv`).

## Live paper trading

Started **2026-10-13 13:31 UTC** with $100,000 virtual capital per account. The
first decisions were for the daily bar closing 2026-10-13 00:00 UTC. `main` opened
30 long positions (45% gross). See `state/status.json`, `state/journal/` and
`reports/live.html`. This sandbox cannot reach exchange APIs, so the engine runs in
archive mode: fills and marks appear once Binance publishes each day (~03:00 UTC).


## Intraday AI day-trading (live, 5-minute bars)

Three AI paper-trading accounts run in **real time** on 12 liquid crypto pairs
(BTC, ETH, SOL, XRP, DOGE, BNB, ADA, AVAX, LINK, LTC, DOT, BCH):

| account | model | trading rule |
|---|---|---|
| `intraday:lgbm_1h` | LightGBM forecasting the next-hour vol-normalised return | trade only extreme forecasts (|z| > 3), hold ≥ 1h, exit on sign flip |
| `intraday:lgbm_4h` | LightGBM forecasting the next 4 hours | enter at |z| > 2.5, hold ≥ 2h, exit on sign flip |
| `intraday:deep_lstm_attn` | LSTM + causal attention network (Momentum-Transformer-lite) | positions straight from the network, trained end-to-end to maximise the Sharpe ratio **net of trading costs** |

**Each 5-minute cycle:**
- **Data:** the daemon pulls the bars that just closed from Binance's public market-data API.
- **Features:** 32 causal features (multi-horizon returns, volatility regime, order-flow imbalance from taker-buy volume, volume surprise, time of day, market and cross-sectional context).
- **Inference:** each model runs and its stateful trading policy advances.
- **Execution:** orders fill at the **live best bid/ask + 1 bp impact + 5 bp taker fee**.
- **Funding:** perpetual funding is charged at OKX's live rates at 00/08/16 UTC.

**Robustness:** each bar is committed atomically (marks, policy state, orders, decisions), so a failure (model, network, database) rolls back and the bar is retried. A bar can never be applied twice or skipped. An adversarial multi-agent review of the live path found five real bugs before launch. All are fixed, and each has a regression test.

```bash
tradinglab intraday-train              # train + freeze models on all history up to the last closed bar
tradinglab intraday-run --commit-every 60   # real-time daemon (commits state/ hourly)
tradinglab intraday-status             # live P&L per AI account
```

**What development research showed** (history was used only to build the models; *evaluation is live*):
- **The forecasts carry information:** the LightGBM next-hour forecasts had a daily rank IC of ≈0.07 out-of-sample from July 2023, versus 0.01 for a linear model.
- **Turning that into net profit is hard at taker fees:** a 1-hour IC of 0.07 is worth only ~4 bps per trade, against a ~12–16 bp round trip.
- **The response:** two accounts trade only rare, extreme forecasts and hold for hours, and the deep model is trained directly on returns after costs.
- **Not backtested:** the exact live configurations were not backtested. Their verdict is the live P&L in `state/intraday_status.json` and `state/intraday.sqlite`. Models are frozen in git (`models/intraday/live/`, with `manifest.json`) before the live data they are judged on.

Live endpoint notes: Binance.com and Binance futures APIs geo-block this cloud region
(HTTP 451), so live features use Binance **spot** 5-minute data (identical source for
training and live). Positions are accounted as perpetuals (shorting, funding).

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
