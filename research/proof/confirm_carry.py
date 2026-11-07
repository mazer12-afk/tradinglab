"""PROTOCOL.md confirmatory tests for the frozen funding-carry rules (research/proof/carry.py).

R2 funding_carry (lookback 14 days, enter above 10 %/yr, leave below 0, at most 1/4 per coin), frozen 7 Nov 2026
R3 always carry BTC + ETH (half each), the parameter-free reference
Design data: the 12 coins, 2020 - 2023. Confirmatory: the 12 coins Jan 2024 - Oct 2026 and the 18 other coins
2020 - Oct 2026 (R3 only on the 12-coin period, being BTC + ETH).

    python research/proof/confirm_carry.py
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
import carry as C  # noqa: E402
from tradinglab.backtest.metrics import deflated_sharpe, probabilistic_sharpe  # noqa: E402

R2 = {"lookback_days": 14, "enter_apr": 0.10, "leave_apr": 0.0, "max_share": 0.25}
SETS = {"12 coins 2024-2026": (C.TWELVE, "2024-01-01", "2026-10-28"), "18 other coins 2020-2026": (C.OTHERS, "2020-01-01", "2026-10-28")}
TRIALS = 12                    # design grid (9) + always-BTC+ETH reference + the two frozen rules


def btc_eth(p: C.CarryPanel) -> pd.DataFrame:
    N = pd.DataFrame(0.0, index=p.grid, columns=p.symbols)
    N[[s for s in ("BTCUSDT", "ETHUSDT") if s in p.symbols]] = 0.5
    return N


def neighbours() -> list[dict]:
    out = []
    for k in ("lookback_days", "enter_apr", "max_share"):
        for f in (0.5, 2.0):
            v = R2[k] * f
            out.append({**R2, k: int(round(v)) if k == "lookback_days" else min(v, 1.0)})
    return out


def run_set(name, syms, lo, hi, rule) -> tuple[dict, pd.Series]:
    p = C.CarryPanel(syms, lo, hi)
    mkt = p.spot_ret.where(p.live).mean(axis=1)
    W = C.funding_carry(p, **R2) if rule == "R2" else btc_eth(p)
    b = C.book(p, W)
    s = C.stats(b, mkt)
    s["coins"] = len(p.symbols)
    s["psr_vs_0"] = round(float(probabilistic_sharpe(b["net"], 0.0)), 4)
    old = (C.SPOT_BPS, C.PERP_BPS)
    C.SPOT_BPS, C.PERP_BPS = 2 * old[0], 2 * old[1]
    s["cost_x2_total_return"] = round(float((1 + C.book(p, W)["net"]).prod() - 1), 4)
    C.SPOT_BPS, C.PERP_BPS = old
    s["neighbours"] = []
    if rule == "R2":
        for nb in neighbours():
            bn = C.book(p, C.funding_carry(p, **nb))
            s["neighbours"].append({"params": nb, "total_return": round(float((1 + bn["net"]).prod() - 1), 4), "sharpe": C.stats(bn)["sharpe"]})
    print(f"{rule} {name}: {s['coins']} coins  Sharpe {s['sharpe']:+.2f}  CAGR {s['cagr']:+.1%}  total {s['total_return']:+.1%}  maxDD {s['max_drawdown']:.1%}  "
          f"funding {s['funding_per_year']:+.1%} basis {s['basis_per_year']:+.1%} cost {s['cost_per_year']:+.1%}  beta {s['beta']:+.3f}  "
          f"alpha t {s['alpha_t']:+.1f}  by year {s['by_year']}")
    return s, b["net"]


def verdict(rule: str, sets: dict, nets: list[pd.Series]) -> dict:
    allr = pd.concat(nets)
    sr = float(allr.mean() / allr.std() * math.sqrt(365))
    years = [v for s in sets.values() for v in s["by_year"].values()]
    checks = {
        "sharpe>=1": sr >= 1.0,
        "psr>=0.99": probabilistic_sharpe(allr, 0.0) >= 0.99,
        "dsr>=0.95": deflated_sharpe(allr, n_trials=TRIALS) >= 0.95,
        "years>=75%": np.mean([y > 0 for y in years]) >= 0.75,
        "all_sets_positive": all(s["total_return"] > 0 for s in sets.values()),
        "maxdd>=-20%": all(s["max_drawdown"] >= -0.20 for s in sets.values()),
        "neighbours_positive": all(nb["total_return"] > 0 for s in sets.values() for nb in s["neighbours"]),
        "cost_x2_positive": all(s["cost_x2_total_return"] > 0 for s in sets.values()),
        "alpha_t>=2_and_|beta|<0.2": all(s["alpha_t"] >= 2 and abs(s["beta"]) < 0.2 for s in sets.values()),
    }
    out = {"rule": rule, "sets": sets, "combined_sharpe": round(sr, 3), "trials_in_family": TRIALS,
           "combined_psr": round(float(probabilistic_sharpe(allr, 0.0)), 4),
           "combined_dsr": round(float(deflated_sharpe(allr, n_trials=TRIALS)), 4),
           "share_of_years_positive": round(float(np.mean([y > 0 for y in years])), 3),
           "checks": {k: bool(v) for k, v in checks.items()}, "proven": bool(all(checks.values()))}
    print(f"{rule} combined: Sharpe {sr:+.2f}  PSR {out['combined_psr']:.4f}  DSR {out['combined_dsr']:.4f}  years positive {out['share_of_years_positive']:.0%}")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"{rule}: {'PROVEN on the backtest tests (forward test still to come)' if out['proven'] else 'NOT PROVEN'}")
    return out


def main() -> None:
    res = {}
    sets, nets = {}, []
    for name, (syms, lo, hi) in SETS.items():
        sets[name], r = run_set(name, syms, lo, hi, "R2")
        nets.append(r)
    res["R2"] = verdict("R2", sets, nets)
    s, r = run_set("BTC+ETH 2024-2026", C.TWELVE, "2024-01-01", "2026-10-28", "R3")
    res["R3"] = verdict("R3", {"BTC+ETH 2024-2026": s}, [r])
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "carry.json").write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
