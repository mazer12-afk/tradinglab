"""Deribit as an independent exchange for the carry rules: hourly funding of every perpetual (full history from the
public API), the index (spot) price each hour, and the perpetual's daily closes.

Day D runs 08:00 UTC of D to 08:00 of D + 1 (Deribit's daily candle): spot_ret and perp_ret over that day;
fund_next = sum of the hourly funding rates inside it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from tradinglab.config import load_config

API = "https://www.deribit.com/api/v2/public/"
ROOT = Path(load_config().paths.data_dir) / "deribit"
ROOT.mkdir(parents=True, exist_ok=True)
END = pd.Timestamp("2026-10-29", tz="UTC")
MIN_DAYS = 30
OFFSET = pd.Timedelta(hours=8)


def _get(s, method, **params):
    for k in range(6):
        try:
            r = s.get(API + method, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()["result"]
            if r.status_code == 429:
                time.sleep(2 + 2 * k)
        except Exception:                          # noqa: BLE001 - retried
            time.sleep(1 + k)
    raise ConnectionError(method)


def perpetuals(s=None) -> list[dict]:
    s = s or requests.Session()
    out = []
    for cur in ("BTC", "ETH", "USDC", "USDT"):
        for i in _get(s, "get_instruments", currency=cur, kind="future", expired="false"):
            if i.get("settlement_period") == "perpetual":
                out.append({"name": i["instrument_name"], "created": int(i["creation_timestamp"]), "kind": i.get("instrument_type") or i.get("future_type")})
    return out


def funding(name: str, created_ms: int, s=None) -> pd.DataFrame:
    """Hourly funding rate and index price of a perpetual since its creation (cached)."""
    path = ROOT / f"funding_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    s = s or requests.Session()
    rows, t = [], max(created_ms, int(pd.Timestamp("2018-08-01", tz="UTC").value // 10**6))
    end = int(END.value // 10**6)
    step = 740 * 3_600_000
    while t < end:
        r = _get(s, "get_funding_rate_history", instrument_name=name, start_timestamp=t, end_timestamp=min(t + step, end))
        rows += r
        t += step
        time.sleep(0.06)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["timestamp", "interest_1h", "index_price"])
    df.index = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df[~df.index.duplicated()].sort_index()[["interest_1h", "index_price"]].astype(float)
    df.to_parquet(path)
    return df


def daily_close(name: str, created_ms: int, s=None) -> pd.Series:
    """Daily closes of the perpetual (candle of day D closes at 00:00 of D + 1)."""
    path = ROOT / f"daily_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)["close"]
    s = s or requests.Session()
    out, t = {}, max(created_ms, int(pd.Timestamp("2018-08-01", tz="UTC").value // 10**6))
    end = int(END.value // 10**6)
    step = 1000 * 86_400_000
    while t < end:
        r = _get(s, "get_tradingview_chart_data", instrument_name=name, start_timestamp=t, end_timestamp=min(t + step, end), resolution="1D")
        for tick, c, v in zip(r.get("ticks", []), r.get("close", []), r.get("volume", [])):
            out[int(tick)] = (float(c), float(v))
        t += step
        time.sleep(0.06)
    ser = pd.DataFrame.from_dict(out, orient="index", columns=["close", "volume"]).sort_index()
    ser.index = pd.to_datetime(ser.index, unit="ms", utc=True)
    ser.to_parquet(path)
    return ser["close"]


class DeribitPanel:
    """What carry.py's rules and book read from a CarryPanel, from Deribit data."""

    def __init__(self, names: list[str] | None = None, start: str | None = None, end: str = "2026-10-28"):
        s = requests.Session()
        meta_path = ROOT / "perpetuals.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps(perpetuals(s)))
        meta = [m for m in json.loads(meta_path.read_text()) if names is None or m["name"] in names]
        sp, pp, fd, first = {}, {}, {}, {}
        for m in meta:
            f = funding(m["name"], m["created"], s)
            if len(f) < 24 * (MIN_DAYS + 5):
                continue
            # Deribit's daily candles run 08:00 - 08:00 UTC: shift every clock by 8 h so "day D" is 08:00 D - 08:00 D+1
            f = f.set_axis(f.index - OFFSET)
            c = daily_close(m["name"], m["created"], s)
            c.index = c.index - OFFSET + pd.Timedelta(days=1)                # price at the end of its day
            idx = f["index_price"]
            idx = idx[idx.index.hour == 0]                                   # index at the day boundary
            day = (f.index - pd.Timedelta(seconds=1)).floor("D")             # an hour ending at t belongs to (D, D + 1]
            fd[m["name"]] = f["interest_1h"].groupby(day).sum()
            sp[m["name"]], pp[m["name"]], first[m["name"]] = idx, c, f.index[0]
        self.symbols = list(sp)
        t0 = pd.Timestamp(start, tz="UTC") if start else min(first.values()).ceil("D")
        self.grid = pd.date_range(t0, pd.Timestamp(end, tz="UTC"), freq="D", inclusive="left")
        S = pd.DataFrame(sp).reindex(self.grid.union(self.grid + pd.Timedelta(days=1)))
        P = pd.DataFrame(pp).reindex(S.index)
        self.spot_ret = (S.shift(-1) / S - 1).reindex(self.grid)
        self.perp_ret = (P.shift(-1) / P - 1).reindex(self.grid)
        self._fund = pd.DataFrame(fd)
        self.fund_next = self._fund.reindex(self.grid).fillna(0.0)
        self.live = pd.DataFrame({n: self.grid >= first[n] + pd.Timedelta(days=MIN_DAYS) for n in self.symbols}, index=self.grid)
        self.live &= self.spot_ret.notna() & self.perp_ret.notna()

    def trailing_funding(self, days: int) -> pd.DataFrame:
        full = pd.date_range(self._fund.index.min(), self.grid[-1], freq="D")
        return self._fund.reindex(full).fillna(0.0).rolling(days, min_periods=1).sum().shift(1).reindex(self.grid) / days
