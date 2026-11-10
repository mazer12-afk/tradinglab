"""The frozen carry rules on an independent exchange (registry 25, 26; pre-registered before the data was loaded).

Deribit perpetuals, spot = Deribit's index price, same engine and costs as on Binance (research/proof/carry.py):
  R3  always carry BTC + ETH, half each           (BTC-PERPETUAL, ETH-PERPETUAL)
  R2  14-day funding > 10 %/yr, leave < 0, <= 1/4  (every Deribit perpetual)

    python research/proof/confirm_deribit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import carry as C  # noqa: E402
import confirm_carry as K  # noqa: E402
from deribit import DeribitPanel  # noqa: E402
from tradinglab.backtest.metrics import probabilistic_sharpe  # noqa: E402


def measure(p, weights_fn, neighbours=()) -> tuple[dict, pd.Series]:
    mkt = p.spot_ret.where(p.live).mean(axis=1)
    W = weights_fn(p)
    b = C.book(p, W)
    first = b.index[(b["notional"] > 0).to_numpy()]
    b = b[b.index >= (first[0] if len(first) else b.index[0])]           # from the first day a position is possible
    s = C.stats(b, mkt)
    s["coins"] = len(p.symbols)
    s["psr_vs_0"] = round(float(probabilistic_sharpe(b["net"], 0.0)), 4)
    old = (C.SPOT_BPS, C.PERP_BPS)
    C.SPOT_BPS, C.PERP_BPS = 2 * old[0], 2 * old[1]
    s["cost_x2_total_return"] = round(float((1 + C.book(p, W)["net"].reindex(b.index)).prod() - 1), 4)
    C.SPOT_BPS, C.PERP_BPS = old
    s["neighbours"] = []
    for nb in neighbours:
        bn = C.book(p, C.funding_carry(p, **nb)).reindex(b.index)
        s["neighbours"].append({"params": nb, "total_return": round(float((1 + bn["net"]).prod() - 1), 4), "sharpe": C.stats(bn)["sharpe"]})
    return s, b["net"]


def show(name, s):
    print(f"{name}: {s['coins']} perpetuals  Sharpe {s['sharpe']:+.2f}  CAGR {s['cagr']:+.1%}  total {s['total_return']:+.1%}  maxDD {s['max_drawdown']:.1%}  "
          f"funding {s['funding_per_year']:+.1%} basis {s['basis_per_year']:+.1%} cost {s['cost_per_year']:+.1%}  beta {s['beta']:+.3f}  alpha t {s['alpha_t']:+.1f}  "
          f"notional {s['avg_notional']:.2f}  by year {s['by_year']}")


def main() -> None:
    res = {}
    p = DeribitPanel(["BTC-PERPETUAL", "ETH-PERPETUAL"])

    def half_each(q):
        W = pd.DataFrame(0.0, index=q.grid, columns=q.symbols)
        W[["BTC-PERPETUAL", "ETH-PERPETUAL"]] = 0.5
        return W.where(q.live.all(axis=1), 0.0)                 # from 30 days after the ETH perpetual's first record

    s, r = measure(p, half_each)
    show("R3 on Deribit (BTC + ETH)", s)
    res["R3"] = K.verdict("R3 on Deribit", {"Deribit BTC + ETH": s}, [r])
    p = DeribitPanel()
    s, r = measure(p, lambda q: C.funding_carry(q, **K.R2), K.neighbours())
    show("R2 on Deribit (all perpetuals)", s)
    res["R2"] = K.verdict("R2 on Deribit", {"Deribit all perpetuals": s}, [r])
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "carry_deribit.json").write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
