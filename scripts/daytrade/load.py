"""Load 1-minute USD-M perp bars for the 12 day-trading coins into aligned frames."""
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.daytrade.sim import Minutes


def load_minutes(start, end) -> Minutes:
    cfg = load_config()
    md = MarketData(cfg.paths.data_dir)
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    f = {c: {} for c in ("open", "high", "low", "close", "volume", "taker_buy_volume")}
    for sym in cfg.intraday.symbols:
        k = md.klines(sym, "1m", "2022-07-01", market="um")
        k = k[(k.index >= s) & (k.index < e)]
        for c in f:
            if c in k:
                f[c][sym] = k[c].astype("float64")
    grid = pd.date_range(s, e, freq="1min", inclusive="left")
    fr = {c: pd.DataFrame(v).reindex(grid) for c, v in f.items()}
    m = Minutes(fr["open"], fr["high"], fr["low"], fr["close"])
    m.volume, m.taker_buy = fr["volume"], fr.get("taker_buy_volume")
    return m
