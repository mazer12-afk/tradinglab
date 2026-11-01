"""The daily forward runner must reproduce the research run on known days (1-27 Oct 2026)."""
import sys, time, logging
from pathlib import Path
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.daytrade import forward as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
cfg = load_config(); md = MarketData(cfg.paths.data_dir)
rec = Path(sys.argv[2]); rec.mkdir(parents=True, exist_ok=True)
seed = pd.read_parquet(sys.argv[1])
seed[seed["t"] < pd.Timestamp("2026-10-01", tz="UTC")].to_parquet(rec / "events.parquet")
t0 = time.time()
for day in pd.date_range("2026-10-01", "2026-10-27", freq="D", tz="UTC"):
    F.trade_day(rec, cfg, md, day)
print(f"{time.time()-t0:.0f}s")
fw = pd.read_csv(rec / "trades.csv")
fw = fw[fw["filled"]]
ref = pd.read_csv("/home/user/tradinglab/reports/daytrade2/test_year_trades.csv")
ref = ref[(pd.to_datetime(ref["entry_time"], utc=True) >= "2026-10-01") & (pd.to_datetime(ref["entry_time"], utc=True) < "2026-10-28")]
key = lambda d: sorted(zip(pd.to_datetime(d["entry_time"], utc=True).dt.strftime("%Y-%m-%d %H:%M"), d["symbol"]))
print("forward filled:", len(fw), "research accepted:", len(ref))
print("forward:", key(fw)); print("research:", key(ref))
# events and forecasts must agree with the seed (computed on full history) for October
ev = pd.read_parquet(rec / "events.parquet")
ev = ev[ev["t"] >= pd.Timestamp("2026-10-01", tz="UTC")]
s = seed[(seed["t"] >= pd.Timestamp("2026-10-01", tz="UTC")) & (seed["t"] < pd.Timestamp("2026-10-28", tz="UTC"))]
j = ev.merge(s, on=["t", "symbol"], suffixes=("_fw", "_res"), how="outer", indicator=True)
print("events: both", int((j._merge == "both").sum()), "only forward", int((j._merge == "left_only").sum()), "only research", int((j._merge == "right_only").sum()))
b = j[j._merge == "both"]
for c in ("z", "doi_z", "sig5_rel", "mkt_3d", "label"):
    print(c, "max abs diff", float((b[c + "_fw"] - b[c + "_res"]).abs().max()))
