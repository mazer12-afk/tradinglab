"""Archive data for day trading: 1-minute USD-M perpetual bars and 5-minute futures metrics."""

from __future__ import annotations

import io

import pandas as pd

from tradinglab.config import utc
from tradinglab.daytrade.sim import Minutes


def minutes(md, symbols: list[str], start, end) -> Minutes:
    """Aligned 1-minute bars (index = minute open, UTC) for [start, end) from the archive cache."""
    s, e = utc(start), utc(end)
    f = {c: {} for c in ("open", "high", "low", "close")}
    for sym in symbols:
        k = md.klines(sym, "1m", s.floor("D").tz_localize(None), market="um")
        k = k[(k.index >= s) & (k.index < e)]
        for c in f:
            f[c][sym] = k[c].astype("float64")
    grid = pd.date_range(s, e, freq="1min", inclusive="left")
    fr = {c: pd.DataFrame(v).reindex(columns=symbols).reindex(grid) for c, v in f.items()}
    return Minutes(fr["open"], fr["high"], fr["low"], fr["close"])


def parse_metrics(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.index = pd.to_datetime(df["create_time"], utc=True)
    df.index.name = "create_time"
    return df.drop(columns=[c for c in ("create_time", "symbol") if c in df]).astype("float64")


def metrics(archive, symbol: str, start, end) -> pd.DataFrame:
    """5-minute futures metrics (open interest, long/short ratios, taker ratio) from the daily archive files."""
    s, e = utc(start).strftime("%Y-%m-%d"), utc(end).strftime("%Y-%m-%d")
    files = [f for f in archive.list_files(f"data/futures/um/daily/metrics/{symbol}/")
             if f.key.endswith(".zip") and s <= f.key[-14:-4] <= e]
    df = archive._load_many(files, parse_metrics)
    return df[~df.index.duplicated(keep="last")].sort_index()


def latest_complete_day(archive, symbols: list[str]):
    """The last UTC day for which every coin has both its 1-minute klines and metrics published."""
    days = []
    for sym in symbols:
        k = [f.key[-14:-4] for f in archive.list_files(f"data/futures/um/daily/klines/{sym}/1m/") if f.key.endswith(".zip")]
        m = [f.key[-14:-4] for f in archive.list_files(f"data/futures/um/daily/metrics/{sym}/") if f.key.endswith(".zip")]
        days.append(min(max(k), max(m)))
    return pd.Timestamp(min(days), tz="UTC")
