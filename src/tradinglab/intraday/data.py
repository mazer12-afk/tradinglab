"""Intraday market data: USD-M perpetual klines (+ premium index, funding) on a regular grid."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradinglab.config import bar_timedelta
from tradinglab.data.store import MarketData, Panel, _funding_on_grid

log = logging.getLogger(__name__)

FIELDS = ["open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume", "trades"]


@dataclass
class IntradayData:
    """Wide frames indexed by bar *close* time (UTC), one column per symbol."""

    bar: str
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    quote_volume: pd.DataFrame
    taker_buy_quote: pd.DataFrame
    trades: pd.DataFrame
    premium: pd.DataFrame
    funding: pd.DataFrame        # funding paid by a long held over (t, t+bar]

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def exec_price(self) -> pd.DataFrame:
        """Fill price for a decision at bar close t: the open of the next bar (starts at t)."""
        return self.open.shift(-1)

    def panel(self) -> Panel:
        return Panel(self.close, self.exec_price(), self.quote_volume, self.funding, self.bar, 0)

    def slice(self, start=None, end=None) -> "IntradayData":
        sl = slice(start, end)
        return IntradayData(self.bar, *(getattr(self, f).loc[sl] for f in
                                         ("open", "high", "low", "close", "quote_volume", "taker_buy_quote",
                                          "trades", "premium", "funding")))


def _regularise(df: pd.DataFrame, grid: pd.DatetimeIndex, max_gap: int = 3) -> pd.DataFrame:
    """Align to a regular grid; forward-fill prices over short gaps, zero volume when missing."""
    df = df.reindex(grid)
    for c in ("close",):
        df[c] = df[c].ffill(limit=max_gap)
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df["close"])
    for c in ("quote_volume", "taker_buy_quote_volume", "trades"):
        if c in df:
            df[c] = df[c].fillna(0.0)
    return df


def load_intraday(md: MarketData, symbols: list[str], start, bar: str = "5m", end=None,
                  default_funding_8h: float = 0.0001, market: str = "spot",
                  extra: dict[str, pd.DataFrame] | None = None) -> IntradayData:
    """Load archived klines (``market`` spot or um) on a regular grid.

    ``extra`` optionally appends more recent klines per symbol (e.g. live bars
    from an exchange API, indexed by open time, same columns).
    """
    td = bar_timedelta(bar)
    with ThreadPoolExecutor(6) as pool:
        kl = dict(zip(symbols, pool.map(lambda s: md.klines(s, bar, start, market=market), symbols)))
        if market == "um":
            pr = dict(zip(symbols, pool.map(lambda s: md.klines(s, bar, start, market="um",
                                                                dataset="premiumIndexKlines"), symbols)))
        else:
            pr = {s: pd.DataFrame() for s in symbols}
        fr = dict(zip(symbols, pool.map(lambda s: md.funding_8h(s, start), symbols)))
    if extra:
        for s, add in extra.items():
            if s in kl and add is not None and not add.empty:
                k = pd.concat([kl[s], add[[c for c in kl[s].columns if c in add.columns]]])
                kl[s] = k[~k.index.duplicated(keep="first")].sort_index()
    d = assemble_intraday(kl, pr, fr, bar, default_funding_8h)
    if end is not None:
        d = d.slice(None, end)
    return d


def assemble_intraday(kl: dict[str, pd.DataFrame], pr: dict[str, pd.DataFrame], fr: dict[str, pd.Series],
                      bar: str = "5m", default_funding_8h: float = 0.0001) -> IntradayData:
    """Wide, regular-grid frames from per-symbol klines (indexed by open time)."""
    td = bar_timedelta(bar)
    symbols = list(kl)
    first = min(k.index[0] for k in kl.values() if not k.empty)
    last = max(k.index[-1] for k in kl.values() if not k.empty)
    grid_open = pd.date_range(first, last, freq=td, tz="UTC")
    frames = {f: {} for f in FIELDS}
    prem = {}
    for s in symbols:
        k = kl[s]
        if k.empty:
            log.warning("no intraday data for %s", s)
            continue
        k = _regularise(k[[c for c in FIELDS if c in k] + ([] if "trades" in k else [])], grid_open)
        for f in FIELDS:
            frames[f][s] = k[f] if f in k else pd.Series(np.nan, index=grid_open)
        p = pr.get(s, pd.DataFrame())
        prem[s] = p["close"].reindex(grid_open).ffill(limit=12) if not p.empty else pd.Series(np.nan, index=grid_open)
    wide = {f: pd.DataFrame(v) for f, v in frames.items()}
    close_index = grid_open + td  # label bars by close time
    for f in wide:
        wide[f].index = close_index
    premium = pd.DataFrame(prem)
    premium.index = close_index
    funding = pd.DataFrame({s: _funding_on_grid(fr.get(s, pd.Series(dtype="float64")), close_index, bar,
                                                pd.Timedelta(0), default_funding_8h)
                            for s in wide["close"].columns})
    return IntradayData(bar, wide["open"], wide["high"], wide["low"], wide["close"], wide["quote_volume"],
                        wide["taker_buy_quote_volume"], wide["trades"], premium, funding)
