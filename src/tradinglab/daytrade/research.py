"""Strategy search with guardrails against false discoveries.

Every variant tried is recorded in a trial registry. A candidate is accepted only if, net of costs,
it made money in each development fold, has a Sharpe ratio of at least ``min_sharpe`` over all of them,
earns at least ``min_net_bp`` per trade, and its Deflated Sharpe Ratio (Bailey & López de Prado), which
discounts for the number of trials made so far, is at least ``min_dsr``. Only accepted candidates are run
on the test period, once each, and every such run is counted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.backtest.metrics import deflated_sharpe
from tradinglab.daytrade.sim import Costs, Minutes, book_stats, funding_cost, run_book, simulate_trades

FOLDS = [("F1", "2022-10-01", "2023-08-01"), ("F2", "2023-08-01", "2024-09-01"), ("F3", "2024-09-01", "2025-10-29")]
TEST = ("2025-10-29", "2026-10-29")


@dataclass
class Rules:
    min_sharpe: float = 1.0
    min_dsr: float = 0.95
    min_net_bp: float = 3.0


def evaluate(trades: pd.DataFrame, rates: dict[str, pd.Series], max_gross: float,
             folds=FOLDS) -> tuple[dict, pd.Series]:
    """Per-fold stats and the pooled daily returns of filled trades (funding added here)."""
    t = trades.copy()
    if "funding" not in t:
        t["funding"] = funding_cost(t, rates)
    out, daily = {}, []
    for name, a, b in folds:
        bk = run_book(t, start=a, end=b, max_gross=max_gross)
        out[name] = book_stats(bk)
        daily.append(bk.daily)
    pooled = pd.concat(daily).sort_index() if daily else pd.Series(dtype=float)
    return out, pooled


class Registry:
    """Append-only log of every trial (one row per variant) and the test runs."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def rows(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    def n_trials(self) -> int:
        r = self.rows()
        return int((r["stage"] == "dev").sum()) if len(r) else 0

    def add(self, family: str, params: dict, folds: dict, pooled: pd.Series, stage: str = "dev",
            rules: Rules = Rules()) -> dict:
        n = self.n_trials() + (1 if stage == "dev" else 0)
        d = pooled.dropna()
        sharpe = float(d.mean() / d.std() * np.sqrt(365)) if len(d) > 10 and d.std() > 0 else float("nan")
        dsr = deflated_sharpe(d, n_trials=max(n, 1)) if len(d) > 10 else float("nan")
        trades = sum(f.get("trades", 0) for f in folds.values())
        net_bp = (sum(f.get("net_bp", 0.0) * f.get("trades", 0) for f in folds.values()) / trades) if trades else float("nan")
        fold_ret = {k: f.get("ann_return", float("nan")) for k, f in folds.items()}
        accepted = bool(stage == "dev" and trades > 0 and all(np.isfinite(v) and v > 0 for v in fold_ret.values())
                        and sharpe >= rules.min_sharpe and dsr >= rules.min_dsr and net_bp >= rules.min_net_bp)
        row = {"stage": stage, "family": family, "params": json.dumps(params, sort_keys=True), "trial_no": n,
               "trades": trades, "sharpe": sharpe, "dsr": dsr, "net_bp": net_bp, "accepted": accepted,
               **{f"ret_{k}": v for k, v in fold_ret.items()},
               **{f"sharpe_{k}": f.get("sharpe", float("nan")) for k, f in folds.items()}}
        r = self.rows()
        pd.concat([r, pd.DataFrame([row])], ignore_index=True).to_csv(self.path, index=False)
        return row


def run_variant(m: Minutes, intents: pd.DataFrame, costs: Costs, rates: dict, max_gross: float, folds=FOLDS):
    tr = simulate_trades(m, intents, costs)
    tr["funding"] = funding_cost(tr, rates)
    return evaluate(tr, rates, max_gross, folds)
