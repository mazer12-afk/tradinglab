"""Where does the cascade-reversal edge come from, and where did it go? (design vs confirmation period)"""
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1])
from load import load_minutes
from tradinglab.daytrade.sim import Costs, simulate_trades, funding_cost
from tradinglab.daytrade.strategies import CascadeReversal, five_minute_z
from tradinglab.config import load_config
from tradinglab.data.store import MarketData

spreads = json.load(open("/home/user/tradinglab/config/sim_spreads.json"))["half_spread"]
costs = Costs(half_spread=spreads)
cfg = load_config(); md = MarketData(cfg.paths.data_dir)
out = []
for lab, a, b, lo in [("A Aug23-Aug24", "2023-08-01", "2024-09-01", "2023-06-15"), ("B Sep24-Oct25", "2024-09-01", "2025-10-29", "2024-07-15")]:
    m = load_minutes(lo, b if lab.startswith("B") else "2024-09-03")
    rates = {s: md.funding_8h(s, pd.Timestamp(lo) - pd.Timedelta(days=30)) for s in m.symbols}
    z, sig, on5 = five_minute_z(m)
    lc = np.log(m.close.ffill())
    mkt_r5 = (lc - lc.shift(5)).mean(axis=1)
    for k in (6.0, 7.0):
        st = CascadeReversal(k=k, entry="limit", offset=1.0, ttl=10, hold=240)
        it = st.intents(m, a, b)
        tr = simulate_trades(m, it, costs)
        tr["funding"] = funding_cost(tr, rates)
        tr = tr[tr["filled"]].copy()
        tr["net"] = tr["gross"] - tr["fees"] - tr["funding"]
        # market-wide vs coin-specific: the market's own 5-minute move in the same sigma units
        mz = (mkt_r5.reindex(tr["t"]).to_numpy() / tr["sigma"].to_numpy())
        tr["market_wide"] = np.abs(mz) > 0.5 * np.abs(tr["z"].to_numpy())
        tr["kind"] = np.where(tr["side"] > 0, "buy the drop", "sell the spike")
        for (kind, mw), g in tr.groupby(["kind", "market_wide"]):
            out.append({"period": lab, "k": k, "side": kind, "move": "market-wide" if mw else "coin-specific",
                        "trades": len(g), "gross_bp": 1e4 * g["gross"].mean(), "net_bp": 1e4 * g["net"].mean(),
                        "win": (g["net"] > 0).mean()})
        if k == 7.0:
            mon = tr.groupby(tr["entry_time"].dt.to_period("M"))["net"].agg(["count", "mean"])
            print(lab, "k=7 monthly net bp:", " ".join(f"{p}:{int(r['count'])}/{1e4*r['mean']:+.0f}" for p, r in mon.iterrows()), flush=True)
t = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(t.round(2).to_string(index=False))
