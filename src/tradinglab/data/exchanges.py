"""Real-time public REST market data from several exchanges.

Every adapter returns *closed* hourly klines indexed by open time (UTC) with the
columns ``open, high, low, close, volume, quote_volume`` and a top-of-book
quote. Adapters are tried in order so the lab keeps working when one venue is
unreachable or geo-blocked (e.g. Binance.com from US cloud regions).

Symbols are Binance-style (``BTCUSDT``); USD venues are mapped to their USD pair,
which tracks the USDT pair to within a few basis points.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import requests

log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume"]


@dataclass
class Quote:
    bid: float
    ask: float
    ts: pd.Timestamp
    venue: str

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


class ExchangeError(RuntimeError):
    pass


def _base(symbol: str) -> str:
    for q in ("USDT", "USDC", "USD"):
        if symbol.endswith(q):
            return symbol[: -len(q)]
    return symbol


class Exchange(ABC):
    name = "exchange"

    def __init__(self, session: requests.Session | None = None, timeout: float = 10.0):
        self.session = session or requests.Session()
        self.timeout = timeout

    def _get(self, url: str, params: dict | None = None):
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code != 200:
            raise ExchangeError(f"{self.name}: HTTP {r.status_code} {r.text[:200]}")
        return r.json()

    @abstractmethod
    def klines_1h(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        """Closed 1h klines with open time >= start."""

    @abstractmethod
    def quote(self, symbol: str) -> Quote:
        """Current best bid/ask."""

    def minute_open(self, symbol: str, ts: pd.Timestamp) -> float | None:
        """Open price of the 1-minute bar starting at ``ts`` (None if unsupported / not yet traded)."""
        return None

    @staticmethod
    def _finish(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
        now = now or pd.Timestamp.now(tz="UTC")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # Drop the still-forming bar.
        return df[df.index + pd.Timedelta(hours=1) <= now][COLUMNS].astype("float64")


class BinanceSpot(Exchange):
    """Binance spot REST (also serves data-api.binance.vision and Binance.US)."""

    def __init__(self, host: str = "https://api.binance.com", name: str = "binance", **kw):
        super().__init__(**kw)
        self.host = host
        self.name = name

    STEP_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}

    def klines(self, symbol: str, interval: str, start: pd.Timestamp, full: bool = False) -> pd.DataFrame:
        """Closed klines with open time >= start. ``full`` keeps trades/taker-buy columns."""
        rows = []
        step = self.STEP_MS[interval]
        cursor = int(pd.Timestamp(start).timestamp() * 1000)
        for _ in range(200):
            data = self._get(f"{self.host}/api/v3/klines",
                             {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": 1000})
            if not data:
                break
            rows.extend(data)
            if len(data) < 1000:
                break
            cursor = int(data[-1][0]) + step
        return self.parse_klines(rows, full=full, bar=pd.Timedelta(milliseconds=step))

    def klines_1h(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        return self.klines(symbol, "1h", start)

    @staticmethod
    def parse_klines(rows: list, full: bool = False, bar: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.DataFrame:
        cols = ["t", "open", "high", "low", "close", "volume", "ct", "quote_volume", "trades",
                "taker_buy_volume", "taker_buy_quote_volume"]
        if not rows:
            return pd.DataFrame(columns=cols[1:] if full else COLUMNS)
        df = pd.DataFrame([r[:11] for r in rows], columns=cols)
        df.index = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
        df.index.name = "open_time"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        now = pd.Timestamp.now(tz="UTC")
        df = df[df.index + bar <= now]  # closed bars only
        keep = ["open", "high", "low", "close", "volume", "quote_volume"]
        if full:
            keep += ["trades", "taker_buy_volume", "taker_buy_quote_volume"]
        return df[keep].astype("float64")

    def minute_open(self, symbol: str, ts: pd.Timestamp) -> float | None:
        t = pd.Timestamp(ts).floor("min")
        data = self._get(f"{self.host}/api/v3/klines",
                         {"symbol": symbol, "interval": "1m", "startTime": int(t.timestamp() * 1000), "limit": 1})
        if data and pd.Timestamp(int(data[0][0]), unit="ms", tz="UTC") - t <= pd.Timedelta(minutes=5):
            return float(data[0][1])
        return None

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        import json as _json

        data = self._get(f"{self.host}/api/v3/ticker/bookTicker", {"symbols": _json.dumps(symbols)})
        now = pd.Timestamp.now(tz="UTC")
        return {d["symbol"]: Quote(float(d["bidPrice"]), float(d["askPrice"]), now, self.name) for d in data}

    def quote(self, symbol: str) -> Quote:
        d = self._get(f"{self.host}/api/v3/ticker/bookTicker", {"symbol": symbol})
        return Quote(float(d["bidPrice"]), float(d["askPrice"]), pd.Timestamp.now(tz="UTC"), self.name)


class Coinbase(Exchange):
    name = "coinbase"
    host = "https://api.exchange.coinbase.com"

    @staticmethod
    def product(symbol: str) -> str:
        return f"{_base(symbol)}-USD"

    def klines_1h(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        frames = []
        cur = pd.Timestamp(start)
        now = pd.Timestamp.now(tz="UTC")
        while cur < now:
            end = min(cur + pd.Timedelta(hours=300), now)
            data = self._get(f"{self.host}/products/{self.product(symbol)}/candles",
                             {"granularity": 3600, "start": cur.isoformat(), "end": end.isoformat()})
            frames.append(self.parse_klines(data))
            cur = end
            time.sleep(0.15)  # public rate limit: 10 req/s
        if not frames:
            return pd.DataFrame(columns=COLUMNS)
        return self._finish(pd.concat(frames))

    @staticmethod
    def parse_klines(data: list) -> pd.DataFrame:
        if not data:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame(data, columns=["t", "low", "high", "open", "close", "volume"])
        df.index = pd.to_datetime(df["t"].astype("int64"), unit="s", utc=True)
        df.index.name = "open_time"
        df["quote_volume"] = df["volume"].astype(float) * df["close"].astype(float)
        return Exchange._finish(df)

    def quote(self, symbol: str) -> Quote:
        d = self._get(f"{self.host}/products/{self.product(symbol)}/ticker")
        return Quote(float(d["bid"]), float(d["ask"]), pd.Timestamp.now(tz="UTC"), self.name)


class Kraken(Exchange):
    name = "kraken"
    host = "https://api.kraken.com"
    ALIASES = {"BTC": "XBT", "DOGE": "XDG"}

    def pair(self, symbol: str) -> str:
        b = _base(symbol)
        return f"{self.ALIASES.get(b, b)}USD"

    def klines_1h(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        # Kraken returns at most the latest 720 candles regardless of `since`.
        data = self._get(f"{self.host}/0/public/OHLC",
                         {"pair": self.pair(symbol), "interval": 60, "since": int(pd.Timestamp(start).timestamp())})
        return self.parse_klines(data)

    @staticmethod
    def parse_klines(data: dict) -> pd.DataFrame:
        if data.get("error"):
            raise ExchangeError(f"kraken: {data['error']}")
        res = {k: v for k, v in data["result"].items() if k != "last"}
        if not res:
            return pd.DataFrame(columns=COLUMNS)
        rows = next(iter(res.values()))
        df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vwap", "volume", "count"])
        df.index = pd.to_datetime(df["t"].astype("int64"), unit="s", utc=True)
        df.index.name = "open_time"
        df["quote_volume"] = df["volume"].astype(float) * df["vwap"].astype(float)
        return Exchange._finish(df)

    def quote(self, symbol: str) -> Quote:
        data = self._get(f"{self.host}/0/public/Ticker", {"pair": self.pair(symbol)})
        if data.get("error"):
            raise ExchangeError(f"kraken: {data['error']}")
        t = next(iter(data["result"].values()))
        return Quote(float(t["b"][0]), float(t["a"][0]), pd.Timestamp.now(tz="UTC"), self.name)


class OKX(Exchange):
    name = "okx"
    host = "https://www.okx.com"

    @staticmethod
    def inst(symbol: str) -> str:
        return f"{_base(symbol)}-USDT"

    def klines_1h(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        frames = []
        after = None
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        for _ in range(40):
            params = {"instId": self.inst(symbol), "bar": "1H", "limit": 100}
            if after:
                params["after"] = after
            data = self._get(f"{self.host}/api/v5/market/history-candles", params)
            rows = data.get("data", [])
            if not rows:
                break
            frames.append(self.parse_klines(data))
            oldest = int(rows[-1][0])
            if oldest <= start_ms:
                break
            after = oldest
            time.sleep(0.1)
        if not frames:
            return pd.DataFrame(columns=COLUMNS)
        df = self._finish(pd.concat(frames))
        return df[df.index >= pd.Timestamp(start)]

    @staticmethod
    def parse_klines(data: dict) -> pd.DataFrame:
        rows = data.get("data", [])
        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame([r[:8] for r in rows],
                          columns=["t", "open", "high", "low", "close", "volume", "vol_ccy", "quote_volume"])
        confirmed = [r[8] == "1" if len(r) > 8 else True for r in rows]
        df = df[confirmed]
        df.index = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
        df.index.name = "open_time"
        return Exchange._finish(df)

    def quote(self, symbol: str) -> Quote:
        d = self._get(f"{self.host}/api/v5/market/ticker", {"instId": self.inst(symbol)})["data"][0]
        return Quote(float(d["bidPx"]), float(d["askPx"]), pd.Timestamp.now(tz="UTC"), self.name)


def default_exchanges(session: requests.Session | None = None) -> list[Exchange]:
    s = session or requests.Session()
    return [
        BinanceSpot("https://api.binance.com", "binance", session=s),
        BinanceSpot("https://data-api.binance.vision", "binance-data", session=s),
        BinanceSpot("https://api.binance.us", "binance-us", session=s),
        OKX(session=s),
        Coinbase(session=s),
        Kraken(session=s),
    ]


def probe(exchanges: list[Exchange], symbol: str = "BTCUSDT") -> list[str]:
    """Names of exchanges answering a quote request right now."""
    ok = []
    for ex in exchanges:
        try:
            q = ex.quote(symbol)
            if q.bid > 0 and q.ask >= q.bid:
                ok.append(ex.name)
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
            log.debug("probe %s failed: %s", ex.name, exc)
    return ok
