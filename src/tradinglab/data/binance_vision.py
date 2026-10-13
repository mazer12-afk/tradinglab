"""Client for Binance's public market-data archive (https://data.binance.vision).

The archive publishes zipped CSVs of every kline interval for every spot and
USD-M futures symbol, as monthly files for completed months and daily files
(published ~2-3h after each UTC midnight) for recent days. Funding-rate history
for perpetuals is published monthly.

This module discovers the available files through the S3 listing API, downloads
them in parallel and caches every parsed file as parquet. Published files are
immutable, so the cache never needs invalidation; only the *set* of files grows.
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

# Download bases, tried in order. The S3 origin is the bucket behind the CDN.
DOWNLOAD_BASES = (
    "https://data.binance.vision",
    "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
)
LIST_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
FLOAT_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_volume", "taker_buy_quote_volume"]


@dataclass(frozen=True)
class ArchiveFile:
    key: str
    last_modified: pd.Timestamp
    size: int

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]


def _to_utc(ts: np.ndarray) -> pd.DatetimeIndex:
    """Binance switched spot archives from ms to µs timestamps in 2025."""
    ts = np.asarray(ts, dtype="int64")
    us = ts > 10**14
    ms = np.where(us, ts // 1000, ts)
    return pd.to_datetime(ms, unit="ms", utc=True)


def parse_kline_csv(raw: bytes) -> pd.DataFrame:
    text = raw.decode()
    has_header = not text[:1].isdigit()
    df = pd.read_csv(io.StringIO(text), header=0 if has_header else None)
    if not has_header:
        df.columns = KLINE_COLUMNS[: df.shape[1]]
    df = df.rename(columns={"count": "trades"})
    out = pd.DataFrame(index=_to_utc(df["open_time"].to_numpy()))
    out.index.name = "open_time"
    for c in FLOAT_COLS:
        if c in df:
            out[c] = df[c].astype("float64").to_numpy()
    if "trades" in df:
        out["trades"] = df["trades"].astype("int64").to_numpy()
    return out[~out.index.duplicated(keep="last")].sort_index()


def parse_funding_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    out = pd.DataFrame(
        {"funding_rate": df["last_funding_rate"].astype("float64").to_numpy()},
        index=_to_utc(df["calc_time"].to_numpy()).floor("min"),
    )
    out.index.name = "time"
    return out[~out.index.duplicated(keep="last")].sort_index()


class BinanceVision:
    """Thread-safe archive client with an on-disk parquet cache."""

    def __init__(self, cache_dir: str | Path, timeout: float = 30.0, max_workers: int = 8,
                 session: requests.Session | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_workers = max_workers
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=max(16, 2 * max_workers))
            session.mount("https://", adapter)
        self._session = session
        self._good_base: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ HTTP
    def _get(self, url: str, retries: int = 4) -> requests.Response:
        delay = 1.0
        last_exc: Exception | None = None
        for _ in range(retries):
            try:
                r = self._session.get(url, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} for {url}")
                return r
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_exc = exc
                time.sleep(delay)
                delay *= 2
        raise ConnectionError(f"GET failed after {retries} attempts: {url}: {last_exc}")

    def list_files(self, prefix: str, start_after: str | None = None) -> list[ArchiveFile]:
        """List archive objects under ``prefix`` (zip files only)."""
        files: list[ArchiveFile] = []
        marker = start_after or ""
        while True:
            url = f"{LIST_BASE}?prefix={prefix}&marker={marker}"
            r = self._get(url)
            r.raise_for_status()
            xml = r.text
            keys = []
            for block in re.findall(r"<Contents>(.*?)</Contents>", xml, flags=re.S):
                key = re.search(r"<Key>(.*?)</Key>", block)
                lm = re.search(r"<LastModified>(.*?)</LastModified>", block)
                size = re.search(r"<Size>(\d+)</Size>", block)
                if not key:
                    continue
                keys.append((key.group(1),))
                if key.group(1).endswith(".zip"):
                    files.append(ArchiveFile(key.group(1), pd.Timestamp(lm.group(1)) if lm else pd.NaT,
                                             int(size.group(1)) if size else 0))
            truncated = "<IsTruncated>true</IsTruncated>" in xml
            if not truncated or not keys:
                break
            marker = keys[-1][0]
        return files

    def fetch(self, key: str) -> bytes:
        bases = [self._good_base] if self._good_base else []
        bases += [b for b in DOWNLOAD_BASES if b != self._good_base]
        errors = []
        for base in bases:
            try:
                r = self._get(f"{base}/{key}", retries=2)
            except ConnectionError as exc:
                errors.append(str(exc))
                continue
            if r.status_code == 200:
                with self._lock:
                    self._good_base = base
                return r.content
            errors.append(f"{base}: HTTP {r.status_code}")
        raise ConnectionError(f"could not download {key}: {errors}")

    @staticmethod
    def _unzip(raw: bytes) -> bytes:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            return z.read(z.namelist()[0])

    def _cached_parse(self, f: ArchiveFile, parser) -> pd.DataFrame:
        path = self.cache_dir / (f.key.replace("/", "__").removesuffix(".zip") + ".parquet")
        if path.exists():
            return pd.read_parquet(path)
        df = parser(self._unzip(self.fetch(f.key)))
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        df.to_parquet(tmp)
        tmp.replace(path)
        return df

    def _load_many(self, files: list[ArchiveFile], parser) -> pd.DataFrame:
        if not files:
            return pd.DataFrame()
        with ThreadPoolExecutor(self.max_workers) as pool:
            frames = list(pool.map(lambda f: self._cached_parse(f, parser), files))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        return df[~df.index.duplicated(keep="last")].sort_index()

    # ------------------------------------------------------------- datasets
    @staticmethod
    def _market_prefix(market: str) -> str:
        return {"spot": "data/spot", "um": "data/futures/um"}[market]

    def kline_files(self, symbol: str, interval: str, start: pd.Timestamp,
                    end: pd.Timestamp | None = None, market: str = "spot",
                    dataset: str = "klines") -> list[ArchiveFile]:
        """Monthly files for full months plus daily files where no monthly exists yet.

        ``dataset`` may be ``klines`` or, for USD-M futures, ``premiumIndexKlines``,
        ``markPriceKlines`` or ``indexPriceKlines`` (same CSV layout).
        """
        mp = self._market_prefix(market)
        start = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start)
        end = (pd.Timestamp(end).tz_localize(None) if end is not None and pd.Timestamp(end).tzinfo
               else (pd.Timestamp(end) if end is not None else None))
        monthly = self.list_files(f"{mp}/monthly/{dataset}/{symbol}/{interval}/")
        month_re = re.compile(r"-(\d{4})-(\d{2})\.zip$")
        covered: set[str] = set()
        chosen: list[ArchiveFile] = []
        for f in monthly:
            m = month_re.search(f.key)
            if not m:
                continue
            mstart = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
            mend = mstart + pd.offsets.MonthEnd(1)
            if mend < start.normalize() or (end is not None and mstart > end):
                continue
            covered.add(f"{m.group(1)}-{m.group(2)}")
            chosen.append(f)
        # Daily files: start listing from the first month not covered by a monthly file.
        last_month = max(covered) if covered else None
        if last_month:
            daily_from = (pd.Timestamp(last_month + "-01") + pd.offsets.MonthBegin(1))
        else:
            daily_from = start.normalize()
        daily_from = max(daily_from, start.normalize())
        marker = (f"{mp}/daily/{dataset}/{symbol}/{interval}/{symbol}-{interval}-"
                  f"{(daily_from - pd.Timedelta(days=1)):%Y-%m-%d}.zip.CHECKSUM")
        daily = self.list_files(f"{mp}/daily/{dataset}/{symbol}/{interval}/", start_after=marker)
        day_re = re.compile(r"-(\d{4}-\d{2}-\d{2})\.zip$")
        for f in daily:
            m = day_re.search(f.key)
            if not m:
                continue
            d = pd.Timestamp(m.group(1))
            if d < daily_from or (end is not None and d > end):
                continue
            if f"{d:%Y-%m}" in covered:
                continue
            chosen.append(f)
        return chosen

    def klines(self, symbol: str, interval: str, start, end=None, market: str = "spot") -> pd.DataFrame:
        files = self.kline_files(symbol, interval, pd.Timestamp(start), end, market)
        df = self._load_many(files, parse_kline_csv)
        if df.empty:
            return df
        start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
        df = df[df.index >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
            df = df[df.index < end_ts]
        return df

    def funding(self, symbol: str, start=None) -> pd.DataFrame:
        files = self.list_files(f"data/futures/um/monthly/fundingRate/{symbol}/")
        if start is not None:
            s = pd.Timestamp(start)
            files = [f for f in files
                     if pd.Timestamp(re.search(r"-(\d{4}-\d{2})\.zip$", f.key).group(1) + "-01")
                     >= (s.tz_localize(None) if s.tzinfo else s).normalize() - pd.offsets.MonthBegin(1)]
        return self._load_many(files, parse_funding_csv)
