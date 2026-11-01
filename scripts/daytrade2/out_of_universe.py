"""The frozen candidate's design, unchanged, on the 18 coins it was never developed on (all periods)."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.daytrade import data as D, events as E
from tradinglab.daytrade.sim import Costs, simulate_trades, funding_cost, run_book, book_stats

cfg = load_config(); md = MarketData(cfg.paths.data_dir)
syms = [s for s in cfg.universe.symbols if s not in cfg.intraday.symbols]
MET = Path(sys.argv[1])
t0 = time.time()
m = D.minutes(md, syms, "2022-07-01", "2026-10-30")
met = {s: pd.read_parquet(MET / f"{s}.parquet") for s in syms}
fund = {s: md.funding_8h(s, pd.Timestamp("2022-06-01")) for s in syms}
ev = E.event_table(m, met, fund)
print("events", len(ev), f"{time.time()-t0:.0f}s", flush=True)
spec = json.load(open("/home/user/tradinglab/config/daytrade2_candidate.json"))["params"]
# the same monthly schedule as the design run (from Feb 2023) and its test year (model at 29 Oct 2025, then monthly)
starts = list(pd.date_range("2023-02-01", "2025-10-01", freq="MS", tz="UTC")) + [pd.Timestamp("2025-10-29", tz="UTC")] + \
         list(pd.date_range("2025-11-01", "2026-10-01", freq="MS", tz="UTC"))
ends = starts[1:] + [pd.Timestamp("2026-11-01", tz="UTC")]
pred = np.full(len(ev), np.nan)
for a, b in zip(starts, ends):
    sel = ((ev["t"] >= a) & (ev["t"] < b)).to_numpy()
    if sel.any():
        pred[sel] = E.predict(E.train(ev, a), ev[sel])
ok = np.isfinite(pred)
s = ev[ok]
print("corr(forecast, outcome):", round(float(np.corrcoef(pred[ok][np.isfinite(s.label)], s.label[np.isfinite(s.label)])[0, 1]), 3), flush=True)
it = E.intents(ev[ok], pred[ok], spec["theta"], spec["hold"], spec["size"])
costs = Costs(half_spread=json.load(open("/home/user/tradinglab/config/sim_spreads.json"))["half_spread"])
tr = simulate_trades(m, it, costs)
tr["funding"] = funding_cost(tr, {s: md.funding_8h(s, pd.Timestamp("2022-08-01")) for s in syms})
rows = []
for name, a, b in [("F1 2023-02..07", "2023-02-01", "2023-08-01"), ("F2 2023-08..2024-08", "2023-08-01", "2024-09-01"),
                   ("F3 2024-09..2025-10", "2024-09-01", "2025-10-29"), ("TEST 2025-10..2026-10", "2025-10-29", "2026-10-29")]:
    st = book_stats(run_book(tr, start=a, end=b, max_gross=spec["max_gross"]))
    rows.append({"period": name, **{k: st.get(k) for k in ("trades", "ann_return", "sharpe", "max_dd", "win_rate", "gross_bp", "net_bp")}})
print(pd.DataFrame(rows).round(3).to_string(index=False))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
