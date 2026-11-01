"""Round 1 of the guarded search: trade-level tests of the metrics-based candidates on all development folds."""
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from load import load_minutes
from signals import event_frame, intents_from_mask, market_hedge
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.daytrade.sim import Costs, simulate_trades, funding_cost
from tradinglab.daytrade.research import FOLDS, Registry, evaluate

REG = Registry(sys.argv[4])
m = load_minutes("2022-09-01", "2025-10-30")
ev = event_frame(m, sys.argv[3])
costs = Costs(half_spread=json.load(open("/home/user/tradinglab/config/sim_spreads.json"))["half_spread"])
md = MarketData(load_config().paths.data_dir)
rates = {s: md.funding_8h(s, pd.Timestamp("2022-08-01")) for s in m.symbols}
hs = np.mean([costs.half_spread.get(s, 1e-4) for s in m.symbols])
hedge_cost = 2 * (costs.taker_bps * 1e-4 + hs + costs.impact_bps * 1e-4)      # market in and out on the basket
in_dev = pd.DataFrame(np.repeat(((ev["grid"] >= pd.Timestamp(FOLDS[0][1], tz="UTC")) & (ev["grid"] < pd.Timestamp(FOLDS[-1][2], tz="UTC")))[:, None], len(m.symbols), axis=1), index=ev["grid"], columns=m.symbols)
z, iz, dz = ev["z"], ev["iz"], ev["doi_z"]
cands = []
for k in (4.0, 5.0):
    for entry, off, ttl in (("market", 0.0, 0), ("limit", 0.5, 15)):
        for hold in (60, 240):
            cands.append(("spike_follow_oi", {"k": k, "entry": entry, "offset": off, "ttl": ttl, "hold": hold, "hedged": False},
                          (z > k) & (dz > 2), 1))
            cands.append(("spike_follow_oi", {"k": k, "entry": entry, "offset": off, "ttl": ttl, "hold": hold, "hedged": True},
                          (z > k) & (dz > 2), 1))
            cands.append(("coin_drop_reversal", {"k": k, "entry": entry, "offset": off, "ttl": ttl, "hold": hold, "hedged": True},
                          iz < -k, 1))
out = []
for fam, p, mask, side in cands:
    it = intents_from_mask(ev, mask & in_dev, side, entry=p["entry"], offset=p["offset"], ttl=p["ttl"], hold=p["hold"])
    it = it.assign(t=pd.DatetimeIndex(it["t"]))
    tr = simulate_trades(m, it, costs)
    tr["funding"] = funding_cost(tr, rates)
    if p["hedged"]:
        tr = market_hedge(tr, m, hedge_cost)
    folds, pooled = evaluate(tr, rates, max_gross=1.0)
    row = REG.add(fam, p, folds, pooled)
    out.append(row)
    print(f"#{row['trial_no']:>3} {fam:20s} {json.dumps(p)}  folds " +
          " ".join(f"{folds[f]['ann_return']:+.0%}" if folds[f].get('trades') else "n/a" for f, _, _ in FOLDS) +
          f"  sharpe {row['sharpe']:.2f} dsr {row['dsr']:.2f} net {row['net_bp']:.1f}bp trades {row['trades']} accepted {row['accepted']}", flush=True)
print("accepted:", sum(r["accepted"] for r in out), "of", len(out), "| registry trials so far:", REG.n_trials())
