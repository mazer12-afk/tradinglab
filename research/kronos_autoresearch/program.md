# Kronos autoresearch

An autonomous research loop in the style of Karpathy's autoresearch: an agent edits one file, runs a fixed
evaluation with a fixed time budget, keeps the change if the score improves and reverts it otherwise, logs every
run, and repeats.

**Goal:** find a trading rule built on Kronos-small's outputs that makes money after costs on the 12 coins.

## Files

- `prepare.py`: fixed. It holds the data (Kronos's cached outputs every 4 hours, market features, labels), the
  walk-forward evaluation, the trading costs and the score. **Do not edit it.**
- `strategy.py`: the only file you edit. It defines `fit(train, emb)` and `predict(model, test, emb)`; anything
  inside them is fair game (features, models, position sizing, filters, smoothing).
- `results.tsv`: one line per run, written by `record.py`.
- `best_strategy.py`: the best version so far (`record.py` keeps it).

## Setup

1. `python research/kronos_autoresearch/prepare.py` builds `data/autoresearch/` from the branch
   `experiment/kronos-cache`.
2. `python research/kronos_autoresearch/strategy.py > run.log 2>&1` runs one evaluation. The first run is the
   baseline.

## The metric

`score` is the annualised Sharpe ratio, after costs, of the 4-hourly book over the **dev months, Jul 2025 – Mar
2026**. Each month the strategy is refitted on every row whose outcome was known before the month began, then
trades the month. Weights are held 4 hours, gross exposure is capped at 1, and each change of weight costs 6 bp
(taker fee plus slippage). Positions held through a settlement pay its funding. Higher is better.

**The holdout, Apr – Oct 2026, is locked.** Do not run `--final` during the loop. It is run once, at the end, on
the best strategy.

**Time budget:** a run must finish within 5 minutes (fits and predictions for all 9 months). A run that exceeds
it, or crashes, counts as a crash.

## The loop

1. Look at `results.tsv` and the current `strategy.py`. Pick one idea.
2. Edit `strategy.py`, then run `python research/kronos_autoresearch/strategy.py > run.log 2>&1`.
3. Run `python research/kronos_autoresearch/record.py "<one-line description>"`. It reads `run.log`, appends to
   `results.tsv`, and either keeps the change (the score beats the best so far: `best_strategy.py` is updated) or
   discards it (`strategy.py` is restored from `best_strategy.py`).
4. If a run crashed, read the end of `run.log`. Fix it if the bug is simple; otherwise give up on the idea.
5. Repeat.

## Judgement

- **Simplicity:** a tiny gain bought with a lot of extra complexity is not worth keeping. Removing something and
  scoring the same is a win.
- **Over-fitting:** every run is one more look at the dev months. Prefer ideas with a reason behind them over
  fine-tuning thresholds. The number of runs is counted, and the final holdout score is what counts.
- **No peeking:** use only what `fit` and `predict` are given. `predict` never sees labels, and must not use one
  test row's information to set the weight of an earlier row.

## Ideas to start from

- Kronos's median 4-hour path, its share of paths up, its spread, high/low and volume forecasts (k_*).
- Its hidden state (`emb`, 512 numbers): reduce it (PCA) and fit a small model to the 4-hour return.
- Cross-sectional (long the best coins, short the worst), market-neutral weights, volatility scaling.
- Lower turnover: hold until the signal flips, smooth the weights, trade only strong signals.
- Mix Kronos's features with simple market features (m_*), or train on the pre-July-2025 rows as well.
