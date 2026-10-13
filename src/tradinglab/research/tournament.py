"""Research tournament: evaluate every candidate out-of-sample and select a portfolio.

Protocol
--------
1. Walk-forward OOS signals for every candidate (deep models retrained each fold).
2. Identical sizing, costs, funding and execution lag for all candidates.
3. Selection on the *selection window* [first_test, holdout_start) using OOS
   Sharpe, Probabilistic Sharpe and Deflated Sharpe (multiple-testing aware),
   plus a correlation filter.
4. The selected ensemble is then evaluated on the untouched *holdout window*
   [holdout_start, end) – the honest estimate of what to expect live.
5. For live trading the same rules are re-applied to the full OOS history and
   trainable winners are refit on all data.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.backtest.engine import BacktestResult, run_backtest
from tradinglab.backtest.metrics import deflated_sharpe, effective_trials, summarize
from tradinglab.config import LabConfig, bars_per_year, utc
from tradinglab.data.store import Panel
from tradinglab.portfolio.allocator import combine_weights, correlation_filter, strategy_alphas
from tradinglab.portfolio.sizing import strategy_weights
from tradinglab.research.walkforward import walk_forward_signals
from tradinglab.strategies.base import Strategy

log = logging.getLogger(__name__)

BENCHMARKS = {"hold_btc", "basket_long"}


@dataclass
class CandidateResult:
    name: str
    description: str
    weights: pd.DataFrame
    backtest: BacktestResult
    metrics: dict = field(default_factory=dict)
    selection_metrics: dict = field(default_factory=dict)
    holdout_metrics: dict = field(default_factory=dict)


def _period(r: pd.Series, start=None, end=None) -> pd.Series:
    if start is not None:
        r = r[r.index >= utc(start)]
    if end is not None:
        r = r[r.index < utc(end)]
    return r


def evaluate_candidate(strategy: Strategy, panel: Panel, cfg: LabConfig, workers: int | None = None,
                       cache_dir: Path | None = None) -> CandidateResult:
    bpy = bars_per_year(cfg.market.bar)
    t0 = time.time()
    sig = walk_forward_signals(strategy, panel, cfg, workers=workers, cache_dir=cache_dir)
    w = strategy_weights(strategy, sig, panel, cfg)
    bt = run_backtest(w, panel, cfg.costs, band=cfg.portfolio.rebalance_band, start=cfg.walkforward.first_test)
    res = CandidateResult(strategy.name, strategy.describe(), w, bt, bt.metrics(bpy))
    log.info("%-12s OOS Sharpe %.2f CAGR %.1f%% MDD %.1f%% (%.0fs)", strategy.name,
             res.metrics.get("sharpe", float("nan")), 100 * res.metrics.get("cagr", float("nan")),
             100 * res.metrics.get("max_drawdown", float("nan")), time.time() - t0)
    return res


def select(results: dict[str, CandidateResult], cfg: LabConfig, start, end,
           include_benchmarks: bool = False) -> tuple[list[str], pd.DataFrame]:
    """Apply the selection rules on returns within [start, end). Returns (chosen, table)."""
    bpy = bars_per_year(cfg.market.bar)
    rets = {n: _period(r.backtest.returns, start, end) for n, r in results.items()}
    per_period_sr = [float(r.mean() / r.std()) if r.std() > 0 else float("nan") for r in rets.values()]
    n_eff = effective_trials(pd.DataFrame(rets).dropna(how="all").fillna(0.0))
    rows = {}
    for n, r in rets.items():
        m = summarize(r, bpy)
        m["dsr"] = deflated_sharpe(r, per_period_sr, n_eff, null_variance=True)
        m["dsr_xs"] = deflated_sharpe(r, per_period_sr, n_eff, null_variance=False)
        m["n_eff_trials"] = n_eff
        rows[n] = m
    table = pd.DataFrame(rows).T
    sc = cfg.selection
    eligible = table[(table["sharpe"] >= sc.min_oos_sharpe) & (table["psr"] >= sc.min_psr)
                     & (table["dsr"] >= sc.min_dsr)]
    if not include_benchmarks:
        eligible = eligible[~eligible.index.isin(BENCHMARKS)]
    ranked = list(eligible.sort_values(["dsr", "sharpe"], ascending=False).index)
    chosen = correlation_filter(rets, ranked, max_corr=0.9, limit=sc.max_strategies)
    table["eligible"] = table.index.isin(eligible.index)
    table["selected"] = table.index.isin(chosen)
    return chosen, table


def ensemble_backtest(results: dict[str, CandidateResult], chosen: list[str], alphas: dict[str, float],
                      panel: Panel, cfg: LabConfig, start=None) -> BacktestResult:
    w = combine_weights({n: results[n].weights for n in chosen}, alphas, panel, cfg)
    return run_backtest(w, panel, cfg.costs, band=cfg.portfolio.rebalance_band, risk=cfg.risk,
                        start=start or cfg.walkforward.first_test)


@dataclass
class TournamentReport:
    created_at: str
    data_end: str
    selection_window: tuple[str, str]
    holdout_window: tuple[str, str]
    candidates: pd.DataFrame
    selection_table: pd.DataFrame
    holdout_table: pd.DataFrame
    holdout_choice: list[str]
    holdout_alphas: dict[str, float]
    holdout_ensemble: dict
    live_choice: list[str]
    live_alphas: dict[str, float]
    live_ensemble_oos: dict
    curves: pd.DataFrame

    def to_json(self) -> dict:
        def clean(d):
            return json.loads(pd.Series(d).to_json()) if isinstance(d, dict) else d
        return {
            "created_at": self.created_at, "data_end": self.data_end,
            "selection_window": self.selection_window, "holdout_window": self.holdout_window,
            "holdout_choice": self.holdout_choice, "holdout_alphas": self.holdout_alphas,
            "holdout_ensemble": clean(self.holdout_ensemble),
            "live_choice": self.live_choice, "live_alphas": self.live_alphas,
            "live_ensemble_oos": clean(self.live_ensemble_oos),
            "candidates": json.loads(self.candidates.to_json(orient="index")),
        }


def run_tournament(strategies: list[Strategy], panel: Panel, cfg: LabConfig, holdout_start: str,
                   workers: int | None = None,
                   cache_dir: Path | None = None) -> tuple[TournamentReport, dict[str, CandidateResult]]:
    bpy = bars_per_year(cfg.market.bar)
    results: dict[str, CandidateResult] = {}
    for s in strategies:
        results[s.name] = evaluate_candidate(s, panel, cfg, workers=workers, cache_dir=cache_dir)

    first_test = cfg.walkforward.first_test
    data_end = str(panel.close.index[-1])
    # 1) selection on the selection window, evaluated on the holdout
    h_choice, sel_table = select(results, cfg, first_test, holdout_start)
    h_alphas = strategy_alphas({n: _period(results[n].backtest.returns, first_test, holdout_start)
                                for n in h_choice}, cfg.selection.weighting)
    h_bt = ensemble_backtest(results, h_choice, h_alphas, panel, cfg, start=holdout_start) if h_choice else None
    h_metrics = h_bt.metrics(bpy) if h_bt is not None else {}
    _, hold_table = select(results, cfg, holdout_start, None)

    # 2) live choice on the full OOS history
    l_choice, _ = select(results, cfg, first_test, None)
    l_alphas = strategy_alphas({n: results[n].backtest.returns for n in l_choice}, cfg.selection.weighting)
    l_bt = ensemble_backtest(results, l_choice, l_alphas, panel, cfg) if l_choice else None

    cand = pd.DataFrame({n: r.metrics for n, r in results.items()}).T
    cand.insert(0, "description", [results[n].description for n in cand.index])
    curves = pd.DataFrame({n: r.backtest.equity for n, r in results.items()})
    if h_bt is not None:
        curves["ENSEMBLE(holdout)"] = h_bt.equity
    if l_bt is not None:
        curves["ENSEMBLE(live rules)"] = l_bt.equity

    report = TournamentReport(
        created_at=pd.Timestamp.now(tz="UTC").isoformat(), data_end=data_end,
        selection_window=(first_test, holdout_start), holdout_window=(holdout_start, data_end),
        candidates=cand, selection_table=sel_table, holdout_table=hold_table,
        holdout_choice=h_choice, holdout_alphas=h_alphas, holdout_ensemble=h_metrics,
        live_choice=l_choice, live_alphas=l_alphas,
        live_ensemble_oos=l_bt.metrics(bpy) if l_bt is not None else {}, curves=curves)
    return report, results


def save_report(report: TournamentReport, results: dict[str, CandidateResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tournament.json").write_text(json.dumps(report.to_json(), indent=2, default=str))
    report.candidates.to_csv(out_dir / "candidates_oos.csv")
    report.selection_table.to_csv(out_dir / "selection_window.csv")
    report.holdout_table.to_csv(out_dir / "holdout_window.csv")
    report.curves.to_csv(out_dir / "equity_curves.csv")
    pd.DataFrame({n: r.backtest.returns for n, r in results.items()}).to_csv(out_dir / "oos_returns.csv")


def fmt_pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"
