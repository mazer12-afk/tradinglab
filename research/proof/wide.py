"""A survivorship-free daily panel of every USDT perpetual Binance ever listed (delisted ones included), with spot
prices where the coin also trades on spot, and funding. Built from the public archive's monthly 1-day files.

Day D's row: ``px`` = price at 00:00 UTC of D (close of the previous daily candle); ``ret`` = return over day D;
``fund`` = funding paid by a long over (D, D + 1]; ``qv`` = quote volume traded during day D - 1 (known at 00:00).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from tradinglab.config import load_config
from tradinglab.data.binance_vision import LIST_BASE, BinanceVision, parse_funding_csv, parse_kline_csv

CFG = load_config()
ROOT = Path(CFG.paths.data_dir) / "wide"
ROOT.mkdir(parents=True, exist_ok=True)
END = pd.Timestamp("2026-11-01", tz="UTC")           # monthly files run through Oct 2026
NOT_COINS = {"BTCDOMUSDT", "USDCUSDT", "DEFIUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT", "ALLUSDT"}   # indices, stablecoin


def _archive() -> BinanceVision:
    return BinanceVision(Path(CFG.paths.data_dir) / "binance_vision")


def list_symbols(market: str, av: BinanceVision | None = None) -> list[str]:
    """Every symbol directory under the market's monthly klines (USDT-quoted)."""
    av = av or _archive()
    prefix = {"um": "data/futures/um/monthly/klines/", "spot": "data/spot/monthly/klines/"}[market]
    out, marker = [], ""
    while True:
        xml = av._get(f"{LIST_BASE}?delimiter=/&prefix={prefix}&marker={marker}").text
        got = re.findall(r"<Prefix>" + re.escape(prefix) + r"([^<]+)/</Prefix>", xml)
        out += got
        if "<IsTruncated>true</IsTruncated>" not in xml or not got:
            break
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
        marker = m.group(1) if m else prefix + got[-1] + "/"
    return sorted({s for s in out if s.endswith("USDT") and s.isascii() and "_" not in s})


def _fetch_all(av: BinanceVision, prefix: str, parser) -> pd.DataFrame:
    files = [f for f in av.list_files(prefix) if f.key.endswith(".zip")]
    if not files:
        return pd.DataFrame()

    def one(f):
        try:
            with zipfile.ZipFile(io.BytesIO(av.fetch(f.key))) as z:
                return parser(z.read(z.namelist()[0]))
        except Exception:                          # noqa: BLE001 - a broken month is skipped
            return pd.DataFrame()

    with ThreadPoolExecutor(8) as pool:
        frames = [x for x in pool.map(one, files) if not x.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def daily(symbol: str, market: str, av: BinanceVision | None = None) -> pd.DataFrame:
    """Daily candles (open, high, low, close, quote_volume) of a symbol, cached in one parquet."""
    path = ROOT / f"{market}_{symbol}_1d.parquet"
    if path.exists():
        return pd.read_parquet(path)
    av = av or _archive()
    mp = {"um": "data/futures/um", "spot": "data/spot"}[market]
    df = _fetch_all(av, f"{mp}/monthly/klines/{symbol}/1d/", parse_kline_csv)
    df = df[["open", "high", "low", "close", "quote_volume"]] if len(df) else pd.DataFrame(columns=["open", "high", "low", "close", "quote_volume"])
    df.to_parquet(path)
    return df


def funding(symbol: str, av: BinanceVision | None = None) -> pd.Series:
    path = ROOT / f"fund_{symbol}.parquet"
    if path.exists():
        return pd.read_parquet(path)["funding_rate"]
    av = av or _archive()
    df = _fetch_all(av, f"data/futures/um/monthly/fundingRate/{symbol}/", parse_funding_csv)
    if df.empty:
        df = pd.DataFrame({"funding_rate": pd.Series(dtype=float)})
    df[["funding_rate"]].to_parquet(path)
    return df["funding_rate"]


def spot_name(perp: str, spot_symbols: set[str]) -> str | None:
    """The spot symbol of a perpetual's coin (1000XXXUSDT and 1MXXXUSDT perps quote XXX in thousands/millions)."""
    for cand in (perp, re.sub(r"^(1000000|1000|1M)", "", perp)):
        if cand in spot_symbols:
            return cand
    return None


def download(log=print) -> dict:
    """Fetch everything (resumable); returns the universe description saved in universe.json."""
    av = _archive()
    um = [s for s in list_symbols("um", av) if s not in NOT_COINS]
    spot = set(list_symbols("spot", av))
    uni = {"perps": um, "spot_of": {s: spot_name(s, spot) for s in um}}
    (ROOT / "universe.json").write_text(json.dumps(uni))
    for i, s in enumerate(um):
        try:
            daily(s, "um", av)
            funding(s, av)
            if uni["spot_of"][s]:
                daily(uni["spot_of"][s], "spot", av)
        except Exception as exc:                   # noqa: BLE001
            log(f"{s}: {type(exc).__name__} {str(exc)[:100]}")
        if (i + 1) % 25 == 0:
            log(f"{i + 1}/{len(um)} symbols")
    return uni


class Wide:
    """The aligned daily panel (index: days at 00:00 UTC; columns: perpetual symbols)."""

    def __init__(self, start: str = "2020-01-01", end: str = "2026-11-01"):
        uni = json.loads((ROOT / "universe.json").read_text())
        days = pd.date_range(pd.Timestamp("2019-09-01", tz="UTC"), END, freq="D")
        px, qv, hi, lo, fund, spx, sqv = {}, {}, {}, {}, {}, {}, {}
        for s in uni["perps"]:
            p = ROOT / f"um_{s}_1d.parquet"
            if not p.exists():
                continue
            k = pd.read_parquet(p)
            k = k[k["quote_volume"] > 0]                              # the archive keeps frozen rows after a delisting
            if len(k) < 5:
                continue
            c = k["close"].astype(float)
            c.index = c.index + pd.Timedelta(days=1)                 # the candle of D - 1 closes at 00:00 of D
            px[s] = c.reindex(days)
            q = k["quote_volume"].astype(float)
            q.index = q.index + pd.Timedelta(days=1)
            qv[s] = q.reindex(days)
            hi[s] = k["high"].astype(float).reindex(days)            # extremes during day D itself
            lo[s] = k["low"].astype(float).reindex(days)
            f = pd.read_parquet(ROOT / f"fund_{s}.parquet")["funding_rate"] if (ROOT / f"fund_{s}.parquet").exists() else pd.Series(dtype=float)
            if len(f):
                idx = pd.DatetimeIndex(f.index)
                idx = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")).round("h")
                # a settlement at time t belongs to the day D with D < t <= D + 1
                fund[s] = pd.Series(f.to_numpy(), index=(idx - pd.Timedelta(seconds=1)).floor("D")).groupby(level=0).sum().reindex(days)
            sp = uni["spot_of"].get(s)
            if sp and (ROOT / f"spot_{sp}_1d.parquet").exists():
                ks = pd.read_parquet(ROOT / f"spot_{sp}_1d.parquet")
                ks = ks[ks["quote_volume"] > 0]
                if len(ks) > 5:
                    cs = ks["close"].astype(float)
                    cs.index = cs.index + pd.Timedelta(days=1)
                    spx[s] = cs.reindex(days)
                    qs = ks["quote_volume"].astype(float)
                    qs.index = qs.index + pd.Timedelta(days=1)
                    sqv[s] = qs.reindex(days)
        sl = slice(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=1))
        self.px_all = pd.DataFrame(px)
        self.symbols = list(self.px_all.columns)
        self.first = self.px_all.apply(lambda c: c.first_valid_index())
        self.last = self.px_all.apply(lambda c: c.last_valid_index())
        self.px = self.px_all.loc[sl]
        self.days = self.px.index
        self.ret = (self.px_all.shift(-1) / self.px_all - 1).loc[sl]
        self.qv = pd.DataFrame(qv).reindex(columns=self.symbols).loc[sl]
        self.qv_all = pd.DataFrame(qv).reindex(columns=self.symbols)
        self.high = pd.DataFrame(hi).reindex(columns=self.symbols).loc[sl]
        self.low = pd.DataFrame(lo).reindex(columns=self.symbols).loc[sl]
        self.fund = pd.DataFrame(fund).reindex(columns=self.symbols).loc[sl]
        self.fund_all = pd.DataFrame(fund).reindex(columns=self.symbols)
        self.spot_all = pd.DataFrame(spx).reindex(columns=self.symbols)
        self.spot_ret = (self.spot_all.shift(-1) / self.spot_all - 1).loc[sl]
        self.spot_qv_all = pd.DataFrame(sqv).reindex(columns=self.symbols)
        # spot is quoted per coin, some perpetuals per 1000 or 1,000,000 coins (ticker prefix)
        self.scale = pd.Series({s: 1e6 if s.startswith(("1000000", "1M")) else 1e3 if s.startswith("1000") else 1.0 for s in self.symbols})
        age = pd.DataFrame({s: (self.days - self.first[s]).days for s in self.symbols}, index=self.days)
        self.age = age.where(age >= 0)
        self.live = self.px.notna() & self.ret.notna()

    def liquidity_rank(self, days: int = 30) -> pd.DataFrame:
        """Rank (1 = most traded) by median daily quote volume over the last ``days`` days, among live coins."""
        v = self.qv_all.rolling(days, min_periods=max(5, days // 3)).median().reindex(self.days)
        return v.where(self.live).rank(axis=1, ascending=False)


if __name__ == "__main__":
    u = download()
    print(len(u["perps"]), "perpetuals;", sum(1 for v in u["spot_of"].values() if v), "with spot")
