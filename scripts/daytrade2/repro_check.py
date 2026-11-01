"""The package code (tradinglab.daytrade.events) must reproduce the frozen candidate's test-year run."""
import json, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.daytrade import data as D, events as E
from tradinglab.daytrade.sim import Costs, simulate_trades, funding_cost, run_book, book_stats

cfg = load_config(); md = MarketData(cfg.paths.data_dir)
syms = cfg.intraday.symbols
t0 = time.time()
m = D.minutes(md, syms, "2022-07-01", "2026-10-30")
met = {s: pd.read_parquet(Path(sys.argv[1]) / f"{s}.parquet") for s in syms}
fund = {s: md.funding_8h(s, pd.Timestamp("2022-06-01")) for s in syms}
ev = E.event_table(m, met, fund)
print("events", len(ev), f"{time.time()-t0:.0f}s", flush=True)
spec = json.load(open("/home/user/tradinglab/config/daytrade2_candidate.json"))["params"]
starts = [pd.Timestamp("2025-10-29", tz="UTC")] + list(pd.date_range("2025-11-01", "2026-10-01", freq="MS", tz="UTC"))
ends = starts[1:] + [pd.Timestamp("2026-11-01", tz="UTC")]
pred = np.full(len(ev), np.nan)
for a, b in zip(starts, ends):
    sel = ((ev["t"] >= a) & (ev["t"] < b)).to_numpy()
    if sel.any():
        pred[sel] = E.predict(E.train(ev, a), ev[sel])
ok = np.isfinite(pred)
it = E.intents(ev[ok], pred[ok], spec["theta"], spec["hold"], spec["size"])
costs = Costs(half_spread=json.load(open("/home/user/tradinglab/config/sim_spreads.json"))["half_spread"])
tr = simulate_trades(m, it, costs)
tr["funding"] = funding_cost(tr, {s: md.funding_8h(s, pd.Timestamp("2022-08-01")) for s in syms})
bk = run_book(tr, start="2025-10-29", end="2026-10-29", max_gross=spec["max_gross"])
st = book_stats(bk)
print({k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()})
ref = pd.read_csv("/home/user/tradinglab/reports/daytrade2/test_year_trades.csv")
key = lambda d: set(zip(pd.to_datetime(d["entry_time"], utc=True).dt.strftime("%Y-%m-%d %H:%M"), d["symbol"], (d["net"] * 1e6).round().astype(int)))
ka, kb = key(bk.trades), key(ref)
print("identical trades:", ka == kb, "| only in package run:", len(ka - kb), "| only in research run:", len(kb - ka))

