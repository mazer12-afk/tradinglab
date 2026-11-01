"""Download Binance USD-M 5-minute futures metrics (open interest, long/short ratios, taker ratio) since 2022-07."""
import io, sys, time
from pathlib import Path
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
a = MarketData(cfg.paths.data_dir).archive


def parse_metrics(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.index = pd.to_datetime(df["create_time"], utc=True)
    df = df.drop(columns=[c for c in ("create_time", "symbol") if c in df])
    return df.astype("float64")


for sym in cfg.intraday.symbols:
    if (OUT / f"{sym}.parquet").exists():
        continue
    t0 = time.time()
    files = [f for f in a.list_files(f"data/futures/um/daily/metrics/{sym}/") if f.key.endswith(".zip")
             and f.key[-14:-4] >= "2022-07-01"]
    df = a._load_many(files, parse_metrics)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(OUT / f"{sym}.parquet")
    print(sym, len(files), "files", len(df), "rows", df.index[0], df.index[-1], f"{time.time()-t0:.0f}s", flush=True)
