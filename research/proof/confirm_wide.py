"""PROTOCOL.md confirmatory test for rules on the survivorship-free daily panel (research/proof/wide.py).

R8 (frozen 10 Nov 2026) - funding-spread book: each day, among the 50 most traded USDT perpetuals (30-day median
volume, point in time, listed >= 30 days), go long the fifth with the lowest funding over the last 3 days and short
the fifth with the highest, equal weights, 0.25 of capital each side (gross 0.5). The signal uses data up to the
previous day's close (a full day of execution delay). Costs: 6 bp per unit traded, plus the funding actually paid
and received. Design data: 1 Jan 2020 - 31 Dec 2023 (registry 20). Confirmatory data: 1 Jan 2024 - 28 Oct 2026.

    python research/proof/confirm_wide.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import widebook as B  # noqa: E402
from wide import Wide  # noqa: E402
from tradinglab.backtest.metrics import deflated_sharpe, probabilistic_sharpe  # noqa: E402

R8 = {"days": 3, "top": 50, "q": 0.2, "gross": 0.5, "lag": 1, "min_age": 30}
TRIALS_FUNDING = 22 + 36 + 1          # earlier funding-factor trials + the wide design grid + R8


def funding_spread(w: Wide, days: int = 3, top: int = 50, q: float = 0.2, gross: float = 0.5, lag: int = 1,
                   min_age: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weights (days x coins) and the tradable-universe mask."""
    uni = (w.liquidity_rank() <= top) & (w.age >= min_age) & w.live
    sig = w.fund_all.rolling(days).sum().shift(1).reindex(w.days).shift(lag).where(uni)    # funding over the days before
    r = sig.rank(axis=1, pct=True)
    long, short = (r <= q), (r > 1 - q)
    W = long.astype(float).div(long.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) \
        - short.astype(float).div(short.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return W * gross / 2, uni


def neighbours() -> list[dict]:
    out = []
    for k, vals in (("days", (2, 6)), ("top", (25, 100)), ("q", (0.1, 0.4))):
        for v in vals:
            out.append({**R8, k: v})
    return out


def evaluate(start: str, end: str, params: dict, cost_bps: float = B.COST_BPS):
    w = Wide("2020-01-01", end)                    # history before ``start`` only feeds the signals
    W, uni = funding_spread(w, **params)
    b = B.perp_book(w, W, cost_bps=cost_bps)
    b = b[b.index >= pd.Timestamp(start, tz="UTC")]
    mkt = w.ret.where(uni).mean(axis=1)
    return b, mkt


def main(start: str = "2024-01-01", end: str = "2026-10-28", name: str = "funding_spread_r8") -> dict:
    b, mkt = evaluate(start, end, R8)
    s = B.stats(b, mkt)
    s["psr_vs_0"] = round(float(probabilistic_sharpe(b["net"], 0.0)), 4)
    s["cost_x2_total_return"] = round(float((1 + evaluate(start, end, R8, 2 * B.COST_BPS)[0]["net"]).prod() - 1), 4)
    s["neighbours"] = []
    for nb in neighbours():
        bn = evaluate(start, end, nb)[0]
        s["neighbours"].append({"params": nb, "total_return": round(float((1 + bn["net"]).prod() - 1), 4), "sharpe": B.stats(bn)["sharpe"]})
    r = b["net"]
    sr = float(r.mean() / r.std() * math.sqrt(365))
    years = list(s["by_year"].values())
    checks = {
        "sharpe>=1": sr >= 1.0,
        "psr>=0.99": probabilistic_sharpe(r, 0.0) >= 0.99,
        "dsr>=0.95": deflated_sharpe(r, n_trials=TRIALS_FUNDING) >= 0.95,
        "years>=75%": np.mean([y > 0 for y in years]) >= 0.75,
        "positive": s["total_return"] > 0,
        "maxdd>=-20%": s["max_drawdown"] >= -0.20,
        "neighbours_positive": all(nb["total_return"] > 0 for nb in s["neighbours"]),
        "cost_x2_positive": s["cost_x2_total_return"] > 0,
        "alpha_t>=2_and_|beta|<0.2": s["alpha_t"] >= 2 and abs(s["beta"]) < 0.2,
    }
    out = {"rule": "R8 funding spread", "params": R8, "period": [start, end], "stats": s, "trials_in_family": TRIALS_FUNDING,
           "psr": round(float(probabilistic_sharpe(r, 0.0)), 4), "dsr": round(float(deflated_sharpe(r, n_trials=TRIALS_FUNDING)), 4),
           "checks": {k: bool(v) for k, v in checks.items()}, "proven": bool(all(checks.values()))}
    print(f"R8 {start} -> {end}: Sharpe {s['sharpe']:+.2f}  CAGR {s['cagr']:+.1%}  total {s['total_return']:+.1%}  maxDD {s['max_drawdown']:.1%}  "
          f"beta {s['beta']:+.2f}  alpha {s['alpha_per_year']:+.1%}/yr (t {s['alpha_t']:+.1f})  funding {s['funding_per_year']:+.1%}  cost {s['cost_per_year']:+.1%}  by year {s['by_year']}")
    print("   neighbours:", [({k: v for k, v in nb["params"].items() if R8[k] != v}, nb["total_return"], nb["sharpe"]) for nb in s["neighbours"]], " cost x2:", s["cost_x2_total_return"])
    print(f"   PSR {out['psr']:.4f}  DSR {out['dsr']:.4f} ({TRIALS_FUNDING} trials)")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("R8:", "PROVEN on the backtest tests (forward test still to come)" if out["proven"] else "NOT PROVEN")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{name}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


if __name__ == "__main__":
    main()
