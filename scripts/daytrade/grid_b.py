"""Cascade reversal on the confirmation period (Sep 2024 → Oct 2025), same grid; selection was fixed on the design period."""
import itertools, json, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1])
from load import load_minutes
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.daytrade.sim import Costs, simulate_trades, run_book, book_stats, funding_cost
from tradinglab.daytrade.strategies import CascadeReversal

A0, A1 = "2024-09-01", "2025-10-29"
m = load_minutes("2024-07-15", "2025-10-29")
spreads = json.load(open("/home/user/tradinglab/config/sim_spreads.json"))["half_spread"]
costs = Costs(half_spread=spreads)
cfg = load_config(); md = MarketData(cfg.paths.data_dir)
rates = {s: md.funding_8h(s, pd.Timestamp("2024-07-01")) for s in m.symbols}
rows = []
t0 = time.time()
import os
if os.environ.get("GRID") == "2":
    grid = list(itertools.product([6.0, 7.0, 8.0], [("limit", 0.5), ("limit", 1.0), ("limit", 1.5)],
                                  [0.0, 1.5, 3.0], [0.0], [120, 240, 480]))
else:
    grid = list(itertools.product([4.0, 5.0, 6.0], [("market", 0.0), ("limit", 0.0), ("limit", 0.5), ("limit", 1.0)],
                                  [0.0, 1.5], [0.0, 3.0], [30, 60, 240]))
for k, (entry, off), tp, sl, hold in grid:
    st = CascadeReversal(k=k, entry=entry, offset=off, ttl=10, tp=tp, sl=sl, hold=hold)
    it = st.intents(m, A0, A1)
    tr = simulate_trades(m, it, costs)
    tr["funding"] = funding_cost(tr, rates)
    b = run_book(tr, start=A0, end=A1)
    s = book_stats(b)
    s.update({"k": k, "entry": entry, "offset": off, "tp": tp, "sl": sl, "hold": hold,
              "fill_rate": float(tr["filled"].mean())})
    rows.append(s)
print(f"{len(grid)} variants in {time.time()-t0:.0f}s")
t = pd.DataFrame(rows)
cols = ["k", "entry", "offset", "tp", "sl", "hold", "trades", "fill_rate", "ann_return", "sharpe", "max_dd", "win_rate", "gross_bp", "cost_bp", "net_bp", "avg_hold_min"]
pd.set_option("display.width", 250, "display.max_rows", 300)
print(t[cols].sort_values("sharpe", ascending=False).round(3).head(30).to_string(index=False))
print("\nshare of variants with positive net return:", float((t["ann_return"] > 0).mean()))
print(t.groupby(["k", "entry", "offset"])[["sharpe", "net_bp"]].median().round(2).to_string())
t.to_csv(sys.argv[2], index=False)
