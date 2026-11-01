"""1-minute USD-M bars and 5-minute futures metrics for the 18 coins outside the day-trading design set."""
import sys, time
from pathlib import Path
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.daytrade.data import metrics

cfg = load_config()
md = MarketData(cfg.paths.data_dir)
new = [s for s in cfg.universe.symbols if s not in cfg.intraday.symbols]
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
print(new, flush=True)
for s in new:
    t0 = time.time()
    try:
        k = md.klines(s, "1m", "2022-07-01", market="um")
        mt = metrics(md.archive, s, "2022-07-01", "2026-10-31")
        mt.to_parquet(OUT / f"{s}.parquet")
        print(s, len(k), str(k.index[0]), len(mt), f"{time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(s, "FAILED", e, flush=True)
