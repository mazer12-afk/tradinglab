"""Round 2: regime-aware event trading (trade with the market's 3-day trend), all development folds."""
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from load import load_minutes
from signals import event_frame, intents_from_mask
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
grid, syms = ev["grid"], m.symbols
lc = np.log(m.close.ffill())
mkt = lc.diff().mean(axis=1).fillna(0.0).cumsum().reindex(grid)
up = (mkt - mkt.shift(288 * 3)) > 0                         # market rose over the previous 3 days
U = pd.DataFrame(np.repeat(up.to_numpy()[:, None], len(syms), axis=1), index=grid, columns=syms)
D = ~U & pd.DataFrame(np.repeat(mkt.shift(288 * 3).notna().to_numpy()[:, None], len(syms), axis=1), index=grid, columns=syms)
in_dev = pd.DataFrame(np.repeat(((grid >= pd.Timestamp(FOLDS[0][1], tz="UTC")) & (grid < pd.Timestamp(FOLDS[-1][2], tz="UTC")))[:, None], len(syms), axis=1), index=grid, columns=syms)
z, dz = ev["z"], ev["doi_z"]
for k in (4.0, 5.0):
    for entry, off, ttl in (("market", 0.0, 0), ("limit", 0.5, 15)):
        for hold in (60, 240):
            longs = U & ((z < -k) | ((z > k) & (dz > 2)))      # uptrend: buy crashes, follow OI-backed spikes
            shorts = D & ((z > k) | ((z < -k) & (dz > 2)))     # downtrend: sell spikes, follow OI-backed crashes
            parts = [intents_from_mask(ev, longs & in_dev, 1, entry, off, ttl, hold),
                     intents_from_mask(ev, shorts & in_dev, -1, entry, off, ttl, hold)]
            it = pd.concat(parts, ignore_index=True)
            it["t"] = pd.DatetimeIndex(it["t"])
            tr = simulate_trades(m, it, costs)
            tr["funding"] = funding_cost(tr, rates)
            folds, pooled = evaluate(tr, rates, max_gross=1.0)
            p = {"k": k, "entry": entry, "offset": off, "ttl": ttl, "hold": hold, "regime": "3d market trend"}
            row = REG.add("regime_events", p, folds, pooled)
            longs_n = int((tr["side"] > 0).sum()); shorts_n = int((tr["side"] < 0).sum())
            print(f"#{row['trial_no']:>3} {json.dumps(p)} folds " +
                  " ".join(f"{folds[f]['ann_return']:+.0%}" if folds[f].get('trades') else "n/a" for f, _, _ in FOLDS) +
                  f" sharpe {row['sharpe']:.2f} dsr {row['dsr']:.2f} net {row['net_bp']:.1f}bp trades {row['trades']} (long {longs_n}/short {shorts_n}) accepted {row['accepted']}", flush=True)
print("registry trials so far:", REG.n_trials())
