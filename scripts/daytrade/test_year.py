"""The single test-year run of the frozen buy-the-crash rule (config/daytrade_frozen.json), plus reference runs."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1])
from load import load_minutes
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.daytrade.sim import Costs, simulate_trades, run_book, book_stats, funding_cost
from tradinglab.daytrade.strategies import CascadeReversal

OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
spec = json.load(open("/home/user/tradinglab/config/daytrade_frozen.json"))
p = spec["params"]
T0, T1 = spec["test"]["period"]
m = load_minutes("2025-09-15", "2026-10-30")
costs = Costs(half_spread=json.load(open("/home/user/tradinglab/config/sim_spreads.json"))["half_spread"])
cfg = load_config(); md = MarketData(cfg.paths.data_dir)
rates = {s: md.funding_8h(s, pd.Timestamp("2025-09-01")) for s in m.symbols}
runs = {
    "buy_the_crash (frozen)": (CascadeReversal(k=p["k"], entry=p["entry"], offset=p["offset"], ttl=p["ttl"], tp=p["tp"], sl=p["sl"],
                                               hold=p["hold"], size=p["size"], sides=p["sides"]), p["max_gross"]),
    "reference: k=7": (CascadeReversal(k=7.0, entry="limit", offset=1.0, ttl=10, hold=240, sides="long"), 1.0),
    "reference: k=6": (CascadeReversal(k=6.0, entry="limit", offset=1.0, ttl=10, hold=240, sides="long"), 1.0),
    "reference: both sides, k=7 (rejected)": (CascadeReversal(k=7.0, entry="limit", offset=1.0, ttl=10, hold=240, sides="both"), 2.0),
}
rows, curves = {}, {}
for name, (st, gross) in runs.items():
    tr = simulate_trades(m, st.intents(m, T0, T1), costs)
    tr["funding"] = funding_cost(tr, rates)
    b = run_book(tr, start=T0, end=T1, max_gross=gross)
    s = book_stats(b)
    s["fill_rate"] = float(tr["filled"].mean())
    s["total_return"] = float((1 + b.daily).prod() - 1)
    rows[name] = s
    curves[name] = (1 + b.daily).cumprod() * 100
    if name.startswith("buy_the_crash"):
        b.trades.to_csv(OUT / "trades.csv", index=False)
        b.equity.to_csv(OUT / "equity.csv")
        mon = b.trades.groupby(b.trades["entry_time"].dt.to_period("M")).agg(trades=("net", "size"), net_bp=("net", lambda x: 1e4 * x.mean()), pnl=("pnl", "sum"))
        print(mon.round(1).to_string())
t = pd.DataFrame(rows).T
pd.set_option("display.width", 250)
print(t[["trades", "fill_rate", "total_return", "ann_return", "sharpe", "max_dd", "win_rate", "gross_bp", "cost_bp", "net_bp", "avg_hold_min"]].round(3).to_string())
t.to_csv(OUT / "summary.csv")
pd.DataFrame(curves).to_csv(OUT / "curves.csv")
