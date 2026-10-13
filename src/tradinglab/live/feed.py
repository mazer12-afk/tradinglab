"""Live market-data feed for the paper-trading engine.

Two modes, chosen automatically (``live.feed: auto``):

* **realtime** – archive history topped up with the latest closed hourly klines
  from public exchange REST APIs; orders fill immediately at the live best
  bid/ask.
* **archive** – only Binance's public archive is reachable (e.g. inside a
  sandbox whose egress policy blocks exchange APIs). Data arrives with the
  archive's daily publication lag (~2-3h after each UTC midnight). Decisions are
  still made in real time with the information available *now*; orders are
  timestamped now and filled at the first 1-minute trade price after
  submission once that minute is published. Results are therefore genuine
  forward-test results on live market prices, revealed with a delay – never
  look-ahead.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from tradinglab.config import utc
from tradinglab.data.exchanges import Exchange, Quote, default_exchanges, probe
from tradinglab.data.store import MarketData, Panel, build_panel, funding_schedule

log = logging.getLogger(__name__)


class LiveFeed:
    def __init__(self, md: MarketData, symbols: list[str], history_start: str, mode: str = "auto",
                 exchanges: list[Exchange] | None = None, default_funding_8h: float = 0.0001):
        self.md = md
        self.symbols = list(symbols)
        self.history_start = history_start
        self.default_funding_8h = default_funding_8h
        self.requested_mode = mode
        self.exchanges = exchanges if exchanges is not None else (default_exchanges() if mode != "archive" else [])
        self.live_exchanges: list[Exchange] = []
        self.mode = "archive"
        self.hourly: dict[str, pd.DataFrame] = {}
        self.rates: dict[str, pd.Series] = {}
        self._closes: pd.DataFrame | None = None
        self._minute_cache: dict[tuple[str, str], pd.DataFrame] = {}

    # ------------------------------------------------------------ refresh
    def resolve_mode(self) -> str:
        if self.requested_mode == "archive":
            self.mode = "archive"
            return self.mode
        names = probe(self.exchanges)
        self.live_exchanges = [e for e in self.exchanges if e.name in names]
        if self.live_exchanges:
            self.mode = "realtime"
        elif self.requested_mode == "realtime":
            raise ConnectionError("realtime feed requested but no exchange API is reachable")
        else:
            self.mode = "archive"
        log.info("feed mode: %s (%s)", self.mode, ", ".join(names) or "no exchange APIs reachable")
        return self.mode

    def refresh(self) -> None:
        self.resolve_mode()
        start = pd.Timestamp(self.history_start)
        with ThreadPoolExecutor(8) as pool:
            hourly = dict(zip(self.symbols, pool.map(lambda s: self.md.hourly(s, start), self.symbols)))
            rates = dict(zip(self.symbols, pool.map(lambda s: self.md.funding_8h(s, start), self.symbols)))
        if self.mode == "realtime":
            for sym in self.symbols:
                hourly[sym] = self._top_up(sym, hourly[sym])
        self.hourly = hourly
        self.rates = rates
        self._closes = None

    def _top_up(self, sym: str, h: pd.DataFrame) -> pd.DataFrame:
        since = (h.index[-1] + pd.Timedelta(hours=1)) if not h.empty else utc(self.history_start)
        for ex in self.live_exchanges:
            try:
                add = ex.klines_1h(sym, since)
            except Exception as exc:  # noqa: BLE001 - try the next venue
                log.debug("%s klines %s failed: %s", ex.name, sym, exc)
                continue
            if add.empty:
                return h
            add = add.assign(n=1)
            out = pd.concat([h, add[[c for c in h.columns if c in add.columns] or add.columns]])
            return out[~out.index.duplicated(keep="first")].sort_index()
        return h

    # -------------------------------------------------------------- prices
    def hourly_closes(self) -> pd.DataFrame:
        """Wide frame of hourly closes indexed by bar *close* time."""
        if self._closes is None:
            frames = {s: h["close"] for s, h in self.hourly.items() if not h.empty}
            c = pd.DataFrame(frames)
            c.index = c.index + pd.Timedelta(hours=1)
            self._closes = c.sort_index()
        return self._closes

    def horizon(self, symbols: list[str] | None = None, active_days: int = 7) -> pd.Timestamp:
        """Latest instant up to which prices of every active symbol are known."""
        closes = self.hourly_closes()
        if closes.empty:
            raise RuntimeError("no market data")
        last_valid = closes.apply(lambda s: s.last_valid_index())
        if symbols is not None:
            last_valid = last_valid.reindex(symbols).dropna()
        newest = last_valid.max()
        # Symbols silent for > active_days are treated as halted/delisted and ignored.
        active = last_valid[last_valid >= newest - pd.Timedelta(days=active_days)]
        return active.min()

    def price_at(self, symbol: str, ts: pd.Timestamp) -> float | None:
        """Last known hourly close at or before ``ts`` (for marking)."""
        c = self.hourly_closes()
        if symbol not in c:
            return None
        s = c[symbol].loc[:ts].dropna()
        return float(s.iloc[-1]) if not s.empty else None

    def fill_price(self, symbol: str, ts: pd.Timestamp) -> float | None:
        """First trade price at/after ``ts`` from 1-minute klines (None until known).

        Realtime mode reads the exchange's live 1-minute klines; archive mode waits
        for the public archive to publish that day.
        """
        ts = utc(ts)
        if self.mode == "realtime" and ts.floor("min") + pd.Timedelta(minutes=1) <= pd.Timestamp.now(tz="UTC"):
            for ex in self.live_exchanges:
                try:
                    px = ex.minute_open(symbol, ts)
                except Exception as exc:  # noqa: BLE001 - try the next venue / the archive
                    log.debug("minute price %s on %s failed: %s", symbol, ex.name, exc)
                    continue
                if px is not None:
                    return px
        day = ts.floor("D")
        key = (symbol, str(day.date()))
        df = self._minute_cache.get(key)
        if df is None:
            try:
                df = self.md.archive.klines(symbol, "1m", day, day + pd.Timedelta(days=1))
            except Exception as exc:  # noqa: BLE001
                log.debug("1m klines %s %s unavailable: %s", symbol, day.date(), exc)
                df = pd.DataFrame()
            if not df.empty and df.index[-1] >= day + pd.Timedelta(hours=23, minutes=59):
                self._minute_cache[key] = df  # complete day → cacheable
        if df is not None and not df.empty:
            after = df[df.index >= ts.floor("min")]
            if not after.empty and after.index[0] - ts.floor("min") <= pd.Timedelta(minutes=5):
                return float(after["open"].iloc[0])
        # Fallback: hourly open of the hour containing ts (only once that hour is known).
        h = self.hourly.get(symbol)
        if h is not None and not h.empty:
            hour = ts.floor("h")
            if hour in h.index and hour + pd.Timedelta(hours=1) <= self.horizon():
                row = h.loc[hour]
                # Linear interpolation inside the hour between open and close.
                frac = (ts - hour) / pd.Timedelta(hours=1)
                return float(row["open"] + frac * (row["close"] - row["open"]))
        return None

    def quote(self, symbol: str) -> Quote | None:
        for ex in self.live_exchanges:
            try:
                q = ex.quote(symbol)
                if q.bid > 0 and q.ask >= q.bid:
                    return q
            except Exception as exc:  # noqa: BLE001
                log.debug("quote %s on %s failed: %s", symbol, ex.name, exc)
        return None

    # ------------------------------------------------------------- funding
    def funding_events(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple]:
        """[(stamp, rate, estimated)] for funding stamps in (start, end]."""
        if end <= start:
            return []
        sched, est = funding_schedule(self.rates.get(symbol, pd.Series(dtype="float64")),
                                      start - pd.Timedelta(days=1), end, self.default_funding_8h)
        sel = (sched.index > start) & (sched.index <= end)
        return [(t, float(r), bool(e)) for t, r, e in zip(sched.index[sel], sched[sel], est[sel])]

    # --------------------------------------------------------------- panel
    def panel(self, bar: str, exec_lag_hours: int, end: pd.Timestamp | None = None) -> Panel:
        hourly = self.hourly
        if end is not None:
            hourly = {s: h[h.index < end] for s, h in hourly.items()}
        return build_panel(hourly, self.rates, bar, exec_lag_hours, self.default_funding_8h)


def latest_bar(horizon: pd.Timestamp, bar: str) -> pd.Timestamp:
    """Close time of the latest bar fully known at ``horizon``."""
    from tradinglab.config import bar_timedelta

    td = bar_timedelta(bar)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return epoch + ((horizon - epoch) // td) * td


def finite(x) -> bool:
    return x is not None and np.isfinite(x)
