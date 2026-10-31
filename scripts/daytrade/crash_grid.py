"""Buy-the-crash (long-only cascade reversal): parameter grid on one period, net of costs."""
import itertools, json, sys, time
import pandas as pd
sys.path.insert(0, sys.argv[1])
from load import load_minutes
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.daytrade.sim import Costs, simulate_trades, run_book, book_stats, funding_cost
from tradinglab.daytrade.strategies import CascadeReversal

P = {"A": ("2023-08-01", "2024-09-01", "2023-06-15", "2024-09-03"), "B": ("2024-09-01", "2025-10-29", "2024-07-15", "2025-10-29")}
a, b, lo, hi = P[sys.argv[2]]
m = load_minutes(lo, hi)
costs = Costs(half_spread=json.load(open("/home/user/tradinglab/config/sim_spreads.json"))["half_spread"])
cfg = load_config(); md = MarketData(cfg.paths.data_dir)
rates = {s: md.funding_8h(s, pd.Timestamp(lo) - pd.Timedelta(days=30)) for s in m.symbols}
rows, t0 = [], time.time()
cache = {}
for k, off, hold, sl in itertools.product([5.0, 6.0, 7.0, 8.0], [0.5, 1.0], [120, 240], [0.0, 6.0]):
    st = CascadeReversal(k=k, entry="limit", offset=off, ttl=10, sl=sl, hold=hold, sides="long")
    tr = simulate_trades(m, st.intents(m, a, b), costs)
    tr["funding"] = funding_cost(tr, rates)
    for gross in (1.0, 2.0):
        s = book_stats(run_book(tr, start=a, end=b, max_gross=gross))
        s.update({"k": k, "offset": off, "hold": hold, "sl": sl, "max_gross": gross, "fill_rate": float(tr["filled"].mean())})
        rows.append(s)
t = pd.DataFrame(rows)
cols = ["k", "offset", "hold", "sl", "max_gross", "trades", "fill_rate", "ann_return", "sharpe", "max_dd", "win_rate", "gross_bp", "cost_bp", "net_bp"]
pd.set_option("display.width", 250, "display.max_rows", 100)
print(f"period {sys.argv[2]} {a} → {b}: {len(t)} variants in {time.time()-t0:.0f}s; positive {float((t.ann_return > 0).mean()):.0%}")
print(t[cols].sort_values("sharpe", ascending=False).round(3).to_string(index=False))
t.to_csv(sys.argv[3], index=False)
