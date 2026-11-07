"""PROTOCOL.md confirmatory tests for the frozen trend-following rule R5 (research/proof/trend.py).

R5 (frozen 7 Nov 2026): trend_weights(lookback 120, kind "scaled", smooth 10, vol_days 30) with the whole book
targeted at 15 % annual volatility (vol_target, 30 days), gross at most 1.
Design data: the 12 coins, 2020 - 2023 (8 settings tried, registry id 8). Confirmatory: the 12 coins Jan 2024 -
Oct 2026, the 18 other coins 2020 - Oct 2026, the 57 fresh coins 2020 - Oct 2026. Trend books take market direction
by design, so the |beta| < 0.2 line (for market-neutral rules) does not apply; alpha t >= 2 does.

    python research/proof/confirm_trend.py
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
import trend as T  # noqa: E402
from tradinglab.backtest.metrics import deflated_sharpe, probabilistic_sharpe  # noqa: E402

R5 = {"lookback": 120, "kind": "scaled", "smooth": 10, "vol_days": 30}
TARGET = 0.15
TRIALS = 9


def weights(p, params):
    return T.vol_target(p, T.trend_weights(p, **params), TARGET)


def neighbours() -> list[dict]:
    out = []
    for k in ("lookback", "smooth", "vol_days"):
        for f in (0.5, 2.0):
            out.append({**R5, k: int(round(R5[k] * f))})
    return out


def main() -> None:
    fresh = json.loads((HERE / "fresh_universe.json").read_text())["symbols"]
    sets = {"12 coins 2024-2026": (T.TWELVE, "2024-01-01", "2026-10-28"),
            "18 other coins 2020-2026": (T.OTHERS, "2020-01-01", "2026-10-28"),
            "57 fresh coins 2020-2026": (fresh, "2020-01-01", "2026-10-28")}
    res, nets = {}, []
    for name, (syms, lo, hi) in sets.items():
        p = T.DailyPanel(syms, lo, hi)
        mkt = T.market(p)
        b = T.book(p, weights(p, R5))
        s = T.stats(b, mkt)
        s["coins"] = len(p.symbols)
        s["cost_x2_total_return"] = round(float((1 + T.book(p, weights(p, R5), cost_bps=2 * T.COST_BPS)["net"]).prod() - 1), 4)
        s["neighbours"] = []
        for nb in neighbours():
            bn = T.book(p, weights(p, nb))
            s["neighbours"].append({"params": nb, "total_return": round(float((1 + bn["net"]).prod() - 1), 4), "sharpe": T.stats(bn)["sharpe"]})
        res[name] = s
        nets.append(b["net"])
        print(f"R5 {name}: {s['coins']} coins  Sharpe {s['sharpe']:+.2f}  CAGR {s['cagr']:+.1%}  total {s['total_return']:+.1%}  maxDD {s['max_drawdown']:.1%}  "
              f"beta {s['beta']:+.2f}  alpha {s['alpha_per_year']:+.1%}/yr (t {s['alpha_t']:+.1f})  cost {s['cost_per_year']:+.1%}  by year {s['by_year']}")
        print("   neighbours:", [(nb["params"]["lookback"], nb["params"]["smooth"], nb["params"]["vol_days"], nb["total_return"]) for nb in s["neighbours"]],
              " cost x2:", s["cost_x2_total_return"])
    allr = pd.concat(nets)
    sr = float(allr.mean() / allr.std() * math.sqrt(365))
    years = [v for s in res.values() for v in s["by_year"].values()]
    checks = {
        "sharpe>=1": sr >= 1.0,
        "psr>=0.99": probabilistic_sharpe(allr, 0.0) >= 0.99,
        "dsr>=0.95": deflated_sharpe(allr, n_trials=TRIALS) >= 0.95,
        "years>=75%": np.mean([y > 0 for y in years]) >= 0.75,
        "all_sets_positive": all(s["total_return"] > 0 for s in res.values()),
        "maxdd>=-20%": all(s["max_drawdown"] >= -0.20 for s in res.values()),
        "neighbours_positive": all(nb["total_return"] > 0 for s in res.values() for nb in s["neighbours"]),
        "cost_x2_positive": all(s["cost_x2_total_return"] > 0 for s in res.values()),
        "alpha_t>=2": all(s["alpha_t"] >= 2 for s in res.values()),
    }
    out = {"rule": "R5", "params": R5, "vol_target": TARGET, "sets": res, "combined_sharpe": round(sr, 3), "trials_in_family": TRIALS,
           "combined_psr": round(float(probabilistic_sharpe(allr, 0.0)), 4), "combined_dsr": round(float(deflated_sharpe(allr, n_trials=TRIALS)), 4),
           "share_of_years_positive": round(float(np.mean([y > 0 for y in years])), 3),
           "checks": {k: bool(v) for k, v in checks.items()}, "proven": bool(all(checks.values()))}
    print(f"R5 combined: Sharpe {sr:+.2f}  PSR {out['combined_psr']:.4f}  DSR {out['combined_dsr']:.4f}  years positive {out['share_of_years_positive']:.0%}")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("R5:", "PROVEN on the backtest tests (forward test still to come)" if out["proven"] else "NOT PROVEN")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "trend_r5.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
