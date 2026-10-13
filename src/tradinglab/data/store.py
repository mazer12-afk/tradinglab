"""Market-data panels built from cached hourly klines.

Conventions (used everywhere in the lab):

* A bar is labelled by its **close time** – the instant its information becomes
  available. The daily bar covering 2026-10-12 is labelled 2026-10-13 00:00 UTC.
* ``close[t]`` is the last trade price of the bar labelled ``t``.
* ``exec_price[t]`` is the price at ``t + exec_lag`` – when orders computed from
  bar ``t`` are filled. Holding-period returns are measured between consecutive
  execution prices, so research results already include the signal→fill delay.
* ``funding[t]`` is the sum of perpetual funding rates paid by a long position
  held over ``(t + lag, t + bar + lag]``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.config import bar_timedelta
from tradinglab.data.binance_vision import BinanceVision, parse_kline_csv

log = logging.getLogger(__name__)


@dataclass
class Panel:
    """Aligned wide frames (index = bar close time, columns = symbols)."""

    close: pd.DataFrame
    exec_price: pd.DataFrame
    volume: pd.DataFrame          # quote-currency volume per bar
    funding: pd.DataFrame
    bar: str
    exec_lag_hours: int

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def returns(self) -> pd.DataFrame:
        """Holding-period return from execution at t to execution at t+1."""
        return self.exec_price.shift(-1) / self.exec_price - 1.0

    def slice(self, start=None, end=None) -> "Panel":
        sl = slice(start, end)
        return Panel(self.close.loc[sl], self.exec_price.loc[sl], self.volume.loc[sl],
                     self.funding.loc[sl], self.bar, self.exec_lag_hours)

    def tail(self, n: int) -> "Panel":
        return Panel(self.close.iloc[-n:], self.exec_price.iloc[-n:], self.volume.iloc[-n:],
                     self.funding.iloc[-n:], self.bar, self.exec_lag_hours)


def hourly_to_bars(h: pd.DataFrame, bar: str) -> pd.DataFrame:
    """Aggregate 1h klines (indexed by open time) into ``bar`` bars labelled by close time."""
    rule = bar[:-1] + {"m": "min", "h": "h", "d": "D", "w": "W"}[bar[-1]]
    agg = h.resample(rule, label="right", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum", "quote_volume": "sum", "n": "sum"})
    # Only keep complete bars.
    expected = bar_timedelta(bar) / pd.Timedelta(hours=1)
    agg = agg[agg["n"] >= expected]
    return agg.drop(columns="n")


class MarketData:
    """Loads, cleans and aligns market data for a universe of symbols."""

    def __init__(self, cache_dir: str | Path, archive: BinanceVision | None = None):
        self.cache_dir = Path(cache_dir)
        self.archive = archive or BinanceVision(self.cache_dir / "binance_vision")
        self.consolidated = self.cache_dir / "consolidated"
        self.consolidated.mkdir(parents=True, exist_ok=True)

    def _klines_1h(self, symbol: str, start) -> pd.DataFrame:
        return self.klines(symbol, "1h", start, market="spot")

    def klines(self, symbol: str, interval: str, start, market: str = "spot",
               dataset: str = "klines") -> pd.DataFrame:
        """All archived klines since ``start`` via a per-symbol consolidated cache.

        Only archive files not yet merged into the consolidated frame are parsed,
        so a refresh after the initial download costs one listing + a few files.
        ``market`` is ``spot`` or ``um`` (USD-M perpetual futures).
        """
        start = pd.Timestamp(start)
        tag = ("" if market == "spot" else f"_{market}") + ("" if dataset == "klines" else f"_{dataset}")
        path = self.consolidated / f"{symbol}_{interval}{tag}.parquet"
        manifest_path = path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        files = self.archive.kline_files(symbol, interval, start, market=market, dataset=dataset)
        cached_start = pd.Timestamp(manifest["start"]) if manifest.get("start") else None
        changed = False
        if cached_start is None or cached_start > start or not path.exists():
            df = self.archive._load_many(files, parse_kline_csv)
            known: set[str] = set()
            changed = True
        else:
            df = pd.read_parquet(path)
            known = set(manifest.get("keys", []))
            new = [f for f in files if f.key not in known]
            if new:
                add = self.archive._load_many(new, parse_kline_csv)
                df = pd.concat([df, add])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                changed = True
        if changed and not df.empty:
            # Unique temp names: several processes may refresh the same cache concurrently.
            tag = f"{os.getpid()}.{threading.get_ident()}"
            tmp = path.with_name(f"{path.stem}.{tag}.tmp")
            df.to_parquet(tmp)
            tmp.replace(path)
            keys = sorted(known | {f.key for f in files})
            mtmp = manifest_path.with_name(f"{manifest_path.stem}.{tag}.tmp")
            mtmp.write_text(json.dumps({"start": str(start.date()), "keys": keys}))
            mtmp.replace(manifest_path)
        start_utc = start.tz_localize("UTC") if start.tzinfo is None else start
        return df[df.index >= start_utc] if not df.empty else df

    def hourly(self, symbol: str, start, end=None, max_jump: float = 0.6) -> pd.DataFrame:
        df = self._klines_1h(symbol, start)
        if end is not None and not df.empty:
            end_ts = pd.Timestamp(end)
            df = df[df.index < (end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts)]
        if df.empty:
            return df
        df = df[["open", "high", "low", "close", "volume", "quote_volume"]].copy()
        df = df[(df["close"] > 0) & (df["open"] > 0)]
        # Drop obviously broken ticks: enormous single-hour round trips.
        lr = np.log(df["close"]).diff()
        bad = (lr.abs() > max_jump) & (lr.shift(-1).abs() > max_jump) & (np.sign(lr) != np.sign(lr.shift(-1)))
        if bad.any():
            log.warning("%s: dropping %d bad ticks", symbol, int(bad.sum()))
            df = df[~bad]
        df["n"] = 1
        return df

    def funding_8h(self, symbol: str, start) -> pd.Series:
        try:
            f = self.archive.funding(symbol, start)
        except ConnectionError as exc:
            log.warning("funding unavailable for %s: %s", symbol, exc)
            return pd.Series(dtype="float64")
        if f.empty:
            return pd.Series(dtype="float64")
        return f["funding_rate"]

    def panel(self, symbols: list[str], bar: str = "1d", start="2018-01-01", end=None,
              exec_lag_hours: int = 0, default_funding_8h: float = 0.0001,
              with_funding: bool = True) -> Panel:
        start_ts = pd.Timestamp(start)
        rates: dict[str, pd.Series] = {}
        with ThreadPoolExecutor(8) as pool:
            hourly = dict(zip(symbols, pool.map(lambda s: self.hourly(s, start_ts, end), symbols)))
            if with_funding:
                rates = dict(zip(symbols, pool.map(lambda s: self.funding_8h(s, start_ts), symbols)))
        return build_panel(hourly, rates if with_funding else None, bar, exec_lag_hours, default_funding_8h)


def build_panel(hourly: dict[str, pd.DataFrame], funding_rates: dict[str, pd.Series] | None,
                bar: str, exec_lag_hours: int, default_funding_8h: float = 0.0001) -> Panel:
    """Assemble a :class:`Panel` from per-symbol 1h klines (shared by research and live)."""
    closes, execs, vols, funds = {}, {}, {}, {}
    lag = pd.Timedelta(hours=exec_lag_hours)
    for sym, h in hourly.items():
        if h is None or h.empty:
            log.warning("no data for %s", sym)
            continue
        if "n" not in h:
            h = h.assign(n=1)
        bars = hourly_to_bars(h, bar)
        if bars.empty:
            continue
        closes[sym] = bars["close"]
        vols[sym] = bars["quote_volume"]
        # Price at t + lag: open of the hourly bar starting then.
        px = h["open"].reindex(bars.index + lag)
        px.index = bars.index
        execs[sym] = px.fillna(bars["close"]) if exec_lag_hours == 0 else px
        if funding_rates is not None:
            funds[sym] = _funding_on_grid(funding_rates.get(sym, pd.Series(dtype="float64")),
                                          bars.index, bar, lag, default_funding_8h)
    close = pd.DataFrame(closes).sort_index()
    order = [s for s in hourly if s in close.columns]
    close = close[order]
    exec_price = pd.DataFrame(execs).reindex(index=close.index, columns=order)
    volume = pd.DataFrame(vols).reindex(index=close.index, columns=order)
    if funding_rates is not None:
        funding = pd.DataFrame(funds).reindex(index=close.index, columns=order).fillna(0.0)
    else:
        funding = pd.DataFrame(0.0, index=close.index, columns=order)
    return Panel(close, exec_price, volume, funding, bar, exec_lag_hours)


def funding_schedule(rates: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                     default_8h: float) -> tuple[pd.Series, pd.Series]:
    """Funding per 8h bucket over [start, end] plus a flag marking estimated buckets.

    Known prints are summed per 8h bucket (some contracts settle every 4h). Before
    the first print the exchange baseline is assumed; after the last archived
    print (the current month is archived only after it ends) the trailing 30-day
    mean is used as the estimate.
    """
    idx = pd.date_range(start.floor("D"), end, freq="8h", tz="UTC" if start.tzinfo is None else None)
    sched = pd.Series(np.nan, index=idx)
    estimated = pd.Series(True, index=idx)
    if rates is not None and not rates.empty:
        r = rates.groupby(rates.index.floor("8h")).sum()
        common = idx.intersection(r.index)
        sched.loc[common] = r.reindex(common)
        estimated.loc[common] = False
        first, last = r.index[0], r.index[-1]
        sched[idx < first] = default_8h
        sched[idx > last] = float(np.clip(r.iloc[-90:].mean(), -0.0005, 0.001))
        sched = sched.ffill().fillna(default_8h)
    else:
        sched[:] = default_8h
    return sched, estimated


def _funding_on_grid(rates: pd.Series, grid: pd.DatetimeIndex, bar: str, lag: pd.Timedelta,
                     default_8h: float) -> pd.Series:
    sched, _ = funding_schedule(rates, grid[0] - pd.Timedelta(days=1),
                                grid[-1] + bar_timedelta(bar) + lag + pd.Timedelta(days=1), default_8h)
    # Sum funding over each holding interval (t+lag, t+bar+lag].
    cum = sched.cumsum()
    start_pts = cum.reindex(grid + lag, method="ffill")
    end_pts = cum.reindex(grid + bar_timedelta(bar) + lag, method="ffill")
    out = pd.Series(end_pts.to_numpy() - start_pts.to_numpy(), index=grid)
    return out.fillna(0.0)
