"""Run the PROTOCOL.md confirmatory tests for a frozen rule (research/proof/rules.py).

    python research/proof/confirm.py lowvol_rv --sets "12:2021-01-01:2024-01-01" "others:2021-01-01:2026-10-28"
Writes research/proof/results/<rule>.json and prints the verdict line by line.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rules  # noqa: E402
import xs  # noqa: E402
from tradinglab.backtest.metrics import deflated_sharpe, probabilistic_sharpe  # noqa: E402


def neighbours(defaults: dict) -> list[dict]:
    out = []
    for k, v in defaults.items():
        for f in (0.5, 2.0):
            nv = type(v)(round(v * f)) if isinstance(v, int) else v * f
            if isinstance(v, float):
                nv = min(nv, 1.0)
            out.append({**defaults, k: nv})
    return out


def trials_in_family(family: str) -> int:
    reg = pd.read_csv(HERE / "registry.csv")
    n = 0
    for r in reg[reg["family"] == family].itertuples():
        n += 15 if r.id == 1 else 1             # the autoresearch loop's 13 ideas and 2 controls
    return max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rule")
    ap.add_argument("--sets", nargs="+", required=True, help="universe:start:end (universe 12 or others)")
    ap.add_argument("--family", default="xs-lowvol")
    a = ap.parse_args()
    spec = rules.RULES[a.rule]
    fn, d = spec["fn"], spec["defaults"]
    commit = subprocess.run(["git", "-C", str(HERE), "log", "-1", "--format=%h", "--", "rules.py"],
                            capture_output=True, text=True).stdout.strip()
    res = {"rule": a.rule, "defaults": d, "rules_commit": commit, "sets": {}}
    nets = []
    for st in a.sets:
        uni, lo, hi = st.split(":")
        syms = xs.TWELVE if uni == "12" else xs.OTHERS
        p = xs.Panel(syms, lo, hi)
        mkt = p.market()
        b = xs.book(p, fn(p, **d))
        s = xs.stats(b, mkt)
        s["coins"] = len(p.symbols)
        s["cost_x2_total_return"] = round(float((1 + xs.book(p, fn(p, **d), cost_bps=2 * xs.COST_BPS)["net"]).prod() - 1), 4)
        s["neighbours"] = []
        for nb in neighbours(d):
            bn = xs.book(p, fn(p, **nb))
            s["neighbours"].append({"params": nb, "total_return": round(float((1 + bn["net"]).prod() - 1), 4),
                                    "sharpe": xs.stats(bn)["sharpe"]})
        s["psr_vs_0"] = round(float(probabilistic_sharpe(b["net"], 0.0)), 4)
        res["sets"][st] = s
        nets.append(b["net"])
        print(f"{st}: {s['coins']} coins  Sharpe {s['sharpe']:+.2f}  return {s['total_return']:+.1%}  CAGR {s['cagr']:+.1%}  "
              f"maxDD {s['max_drawdown']:.1%}  beta {s['beta']:+.2f}  alpha {s['alpha_per_year']:+.1%}/yr (t {s['alpha_t']:+.1f})  "
              f"by year {s['by_year']}")
    allr = pd.concat(nets)
    n = trials_in_family(a.family)
    sr = float(allr.mean() / allr.std() * math.sqrt(xs.PER_YEAR))
    years = [v for s in res["sets"].values() for v in s["by_year"].values()]
    checks = {
        "sharpe>=1": sr >= 1.0,
        "psr>=0.99": probabilistic_sharpe(allr, 0.0) >= 0.99,
        "dsr>=0.95": deflated_sharpe(allr, n_trials=n) >= 0.95,
        "years>=75%": np.mean([y > 0 for y in years]) >= 0.75,
        "both_sets_positive": all(s["total_return"] > 0 for s in res["sets"].values()),
        "maxdd>=-20%": all(s["max_drawdown"] >= -0.20 for s in res["sets"].values()),
        "neighbours_positive": all(nb["total_return"] > 0 for s in res["sets"].values() for nb in s["neighbours"]),
        "cost_x2_positive": all(s["cost_x2_total_return"] > 0 for s in res["sets"].values()),
        "alpha_t>=2_and_|beta|<0.2": all(s["alpha_t"] >= 2 and abs(s["beta"]) < 0.2 for s in res["sets"].values()),
    }
    res.update({"combined_sharpe": round(sr, 3), "trials_in_family": n,
                "combined_psr": round(float(probabilistic_sharpe(allr, 0.0)), 4),
                "combined_dsr": round(float(deflated_sharpe(allr, n_trials=n)), 4),
                "share_of_years_positive": round(float(np.mean([y > 0 for y in years])), 3),
                "checks": {k: bool(v) for k, v in checks.items()}, "proven": bool(all(checks.values()))})
    print(f"combined: Sharpe {sr:+.2f}  PSR {res['combined_psr']:.3f}  DSR {res['combined_dsr']:.3f} ({n} trials)  "
          f"years positive {res['share_of_years_positive']:.0%}")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("PROVEN" if res["proven"] else "NOT PROVEN")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{a.rule}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
