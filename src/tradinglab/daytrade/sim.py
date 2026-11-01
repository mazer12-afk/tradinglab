"""Day-trading simulator: individual trades on 1-minute bars, with market or limit orders.

A strategy emits trade *intents*, each decided at the close of minute ``s`` from data up to that close:
coin, side (+1 long, -1 short), size (fraction of equity), entry order (market, or a limit valid for
``ttl`` minutes) and exit rules (take-profit, stop, maximum holding time). Fills are conservative:

* a market order fills at the next minute's open, plus the coin's half-spread and ``impact``, and pays
  the taker fee;
* a limit order fills at its price only if a later minute trades *through* it by ``through_bp`` (a buy
  limit needs a low below it; queue position at the exact price is never assumed), and pays the maker
  fee; unfilled after ``ttl`` minutes, it is cancelled;
* the take-profit rests as a limit order (maker, same trade-through rule); the stop is a market order
  triggered when the minute's range reaches it, filled at the stop or at the open if the price gapped
  through it; when both are reached in the same minute, the stop is assumed to come first;
* the time exit is a market order at the open of the minute ``max_hold`` minutes after the fill;
* exit checks start in the fill minute for a market entry (filled at its open) and in the next minute
  for a limit entry (the order of prices inside the fill minute is unknown);
* funding is paid or received for every settlement (00/08/16 UTC) the position is held through.

Portfolio: one open trade per coin and strategy; a trade is skipped when it would take the open
notional above ``max_gross`` × equity. Equity changes when trades close (day trades are short, so
marks in between are not needed for daily returns).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradinglab.config import utc

MARKET, LIMIT = 0, 1
EXIT_NONE, EXIT_TP, EXIT_SL, EXIT_TIME, EXIT_END = 0, 1, 2, 3, 4


@dataclass
class Costs:
    maker_bps: float = 2.0          # Binance USD-M VIP0 maker fee
    taker_bps: float = 5.0          # Binance USD-M VIP0 taker fee
    impact_bps: float = 1.0         # extra cost of a market order beyond the half-spread
    through_bps: float = 1.0        # a limit fills only if the price trades this far through it
    half_spread: dict = field(default_factory=dict)   # per coin, fraction of price (default 1 bp)


@dataclass
class Minutes:
    """Aligned 1-minute bars: index = minute open time (UTC), one column per coin."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def slice(self, start=None, end=None) -> "Minutes":
        sl = slice(start, end)
        return Minutes(self.open.loc[sl], self.high.loc[sl], self.low.loc[sl], self.close.loc[sl])


def minutes_from_klines(kl: dict[str, pd.DataFrame]) -> Minutes:
    idx = pd.DatetimeIndex(sorted(set().union(*[k.index for k in kl.values()])))
    grid = pd.date_range(idx[0], idx[-1], freq="1min")
    f = {c: pd.DataFrame({s: k[c].reindex(grid) for s, k in kl.items()}) for c in ("open", "high", "low", "close")}
    return Minutes(f["open"], f["high"], f["low"], f["close"])


def _simulate_py(o, h, lo, c, s_idx, coin, side, etype, eprice, ttl, tp, sl, max_hold,
                 maker, taker, slip, through):
    """Fill every intent independently → entry/exit minute, prices, costs and exit kind."""
    n = len(s_idx)
    T = o.shape[0]
    e_i = np.full(n, -1, np.int64)
    x_i = np.full(n, -1, np.int64)
    e_px = np.full(n, np.nan)
    x_px = np.full(n, np.nan)
    cost = np.zeros(n)
    kind = np.zeros(n, np.int64)
    for k in range(n):
        j, s, d = coin[k], s_idx[k], side[k]
        f = -1
        px = np.nan
        if etype[k] == MARKET:
            m = s + 1
            while m < T and np.isnan(o[m, j]):
                m += 1
            if m >= T:
                continue
            f = m
            px = o[m, j] * (1.0 + d * slip[j])
            cost[k] += taker
            first = f
        else:
            lim = eprice[k]
            for m in range(s + 1, min(s + 1 + ttl[k], T)):
                if np.isnan(lo[m, j]):
                    continue
                if (d > 0 and lo[m, j] < lim * (1.0 - through)) or (d < 0 and h[m, j] > lim * (1.0 + through)):
                    f = m
                    # a limit is filled at its price, or at the open if the minute opened beyond it
                    px = min(lim, o[m, j]) if d > 0 else max(lim, o[m, j])
                    break
            if f < 0:
                continue
            cost[k] += maker
            first = f + 1
        e_i[k] = f
        e_px[k] = px
        tp_px = px * (1.0 + d * tp[k]) if tp[k] > 0 else np.nan
        sl_px = px * (1.0 - d * sl[k]) if sl[k] > 0 else np.nan
        last = min(f + max_hold[k], T - 1)
        if last < first:             # filled in the last minute of the data: nothing to close at
            e_i[k] = -1
            continue
        done = False
        for m in range(first, last):
            if np.isnan(lo[m, j]):
                continue
            if sl[k] > 0 and ((d > 0 and lo[m, j] <= sl_px) or (d < 0 and h[m, j] >= sl_px)):
                gap = o[m, j] if m > first or etype[k] == LIMIT else sl_px
                fill = min(sl_px, gap) if d > 0 else max(sl_px, gap)
                x_i[k], x_px[k], kind[k] = m, fill * (1.0 - d * slip[j]), EXIT_SL
                cost[k] += taker
                done = True
                break
            if tp[k] > 0 and ((d > 0 and h[m, j] > tp_px * (1.0 + through)) or (d < 0 and lo[m, j] < tp_px * (1.0 - through))):
                fill = max(tp_px, o[m, j]) if d > 0 else min(tp_px, o[m, j])
                x_i[k], x_px[k], kind[k] = m, fill, EXIT_TP
                cost[k] += maker
                done = True
                break
        if not done:
            m = last
            while m < T - 1 and np.isnan(o[m, j]):
                m += 1
            if np.isnan(o[m, j]):
                e_i[k] = -1          # no price to close at: drop the trade (data end)
                continue
            x_i[k], x_px[k] = m, o[m, j] * (1.0 - d * slip[j])
            kind[k] = EXIT_TIME if m == f + max_hold[k] else EXIT_END
            cost[k] += taker
    return e_i, x_i, e_px, x_px, cost, kind


_SIM = None


def _sim():
    global _SIM
    if _SIM is None:
        try:
            import numba

            _SIM = numba.njit(cache=True)(_simulate_py)
        except Exception:  # noqa: BLE001
            _SIM = _simulate_py
    return _SIM


def simulate_trades(m: Minutes, intents: pd.DataFrame, costs: Costs) -> pd.DataFrame:
    """Fill ``intents`` (columns: t, symbol, side, size, entry ['market'|'limit'], price, ttl, tp, sl, max_hold;
    tp/sl as fractions of the entry price, 0 = none; ``t`` = the minute whose close the decision used)."""
    if intents.empty:
        return intents.assign(filled=False)
    cols = {s: i for i, s in enumerate(m.symbols)}
    s_idx = m.index.get_indexer(pd.DatetimeIndex(intents["t"]))
    if (s_idx < 0).any():
        raise ValueError("intent times must be minutes of the data")
    slip = np.array([costs.half_spread.get(s, 1e-4) + costs.impact_bps * 1e-4 for s in m.symbols])
    out = _sim()(m.open.to_numpy("float64"), m.high.to_numpy("float64"), m.low.to_numpy("float64"),
                 m.close.to_numpy("float64"), s_idx.astype(np.int64),
                 intents["symbol"].map(cols).to_numpy(np.int64), intents["side"].to_numpy(np.int64),
                 (intents["entry"] == "limit").to_numpy(np.int64), intents["price"].fillna(0.0).to_numpy("float64"),
                 intents["ttl"].fillna(0).to_numpy(np.int64), intents["tp"].fillna(0.0).to_numpy("float64"),
                 intents["sl"].fillna(0.0).to_numpy("float64"), intents["max_hold"].to_numpy(np.int64),
                 costs.maker_bps * 1e-4, costs.taker_bps * 1e-4, slip, costs.through_bps * 1e-4)
    e_i, x_i, e_px, x_px, cost, kind = out
    res = intents.copy()
    ok = e_i >= 0
    res["filled"] = ok
    res["entry_time"] = pd.Series(m.index.take(np.where(ok, e_i, 0)), index=res.index).where(ok)
    res["exit_time"] = pd.Series(m.index.take(np.where(ok, x_i, 0)), index=res.index).where(ok)
    res["entry_px"], res["exit_px"], res["fees"] = e_px, x_px, cost
    res["exit_kind"] = pd.Series(kind, index=res.index).map(
        {EXIT_NONE: "", EXIT_TP: "take-profit", EXIT_SL: "stop", EXIT_TIME: "time", EXIT_END: "data end"})
    res["gross"] = res["side"] * (res["exit_px"] / res["entry_px"] - 1.0)
    return res


def funding_cost(trades: pd.DataFrame, rates: dict[str, pd.Series]) -> np.ndarray:
    """Funding paid (+) per unit notional for each filled trade: side × rate at each settlement held through."""
    out = np.zeros(len(trades))
    for k, (sym, d, a, b) in enumerate(zip(trades["symbol"], trades["side"], trades["entry_time"], trades["exit_time"])):
        if pd.isna(a):
            continue
        r = rates.get(sym)
        if r is None or r.empty:
            continue
        sel = r[(r.index > a) & (r.index <= b)]
        out[k] = d * float(sel.sum())
    return out


@dataclass
class Book:
    trades: pd.DataFrame          # accepted trades with pnl
    daily: pd.Series              # daily returns on realised equity
    equity: pd.Series             # realised equity after each day


def run_book(trades: pd.DataFrame, capital: float = 100_000.0, max_gross: float = 2.0,
             start=None, end=None) -> Book:
    """Accept filled trades in entry order (one per coin at a time, gross cap), compound realised P&L."""
    t = trades[trades["filled"]].copy()
    if "priority" in t:              # same-minute entries: the strongest signal is taken first
        t = t.assign(_p=-t["priority"]).sort_values(["entry_time", "_p", "symbol"]).drop(columns="_p")
    else:
        t = t.sort_values(["entry_time", "symbol"])
    if start is not None:
        t = t[t["entry_time"] >= utc(start)]
    if end is not None:
        t = t[t["exit_time"] < utc(end)]
    t["net"] = t["gross"] - t["fees"] - t.get("funding", 0.0)
    eq = capital
    busy: set[str] = set()
    keep, notional, pnl = [], [], []
    heap: list = []                # open trades: (exit_time, id, symbol, notional, net return)
    for row in t.itertuples():
        while heap and heap[0][0] <= row.entry_time:
            xt, _, sym, nt, r = heapq.heappop(heap)
            eq += nt * r
            busy.discard(sym)
        if row.symbol in busy or row.size <= 0:
            keep.append(False); notional.append(0.0); pnl.append(0.0)
            continue
        nt = row.size * eq
        gross_open = sum(x[3] for x in heap)
        if gross_open + nt > max_gross * eq:
            keep.append(False); notional.append(0.0); pnl.append(0.0)
            continue
        heapq.heappush(heap, (row.exit_time, row.Index, row.symbol, nt, row.net))
        busy.add(row.symbol)
        keep.append(True); notional.append(nt); pnl.append(nt * row.net)
    t["accepted"], t["notional"], t["pnl"] = keep, notional, pnl
    acc = t[t["accepted"]]
    if acc.empty:
        return Book(acc, pd.Series(dtype=float), pd.Series(dtype=float))
    by_day = acc.groupby(acc["exit_time"].dt.floor("D"))["pnl"].sum()

    def day(x):
        x = pd.Timestamp(x)
        return (x.tz_localize("UTC") if x.tzinfo is None else x).floor("D")

    days = pd.date_range(by_day.index[0] if start is None else day(start),
                         by_day.index[-1] if end is None else day(end) - pd.Timedelta(days=1), freq="D")
    pnl_day = by_day.reindex(days, fill_value=0.0)
    equity = capital + pnl_day.cumsum()
    daily = pnl_day / equity.shift(1).fillna(capital)
    return Book(acc, daily, equity)


def book_stats(b: Book) -> dict:
    d = b.daily
    if d.empty:
        return {"trades": 0}
    eq = (1 + d).cumprod()
    t = b.trades
    yrs = len(d) / 365
    return {"trades": int(len(t)), "trades_per_day": len(t) / len(d), "ann_return": eq.iloc[-1] ** (1 / yrs) - 1,
            "sharpe": d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan,
            "max_dd": float((eq / eq.cummax() - 1).min()), "win_rate": float((t["net"] > 0).mean()),
            "gross_bp": float(1e4 * t["gross"].mean()), "net_bp": float(1e4 * t["net"].mean()),
            "cost_bp": float(1e4 * (t["fees"] + t.get("funding", 0.0)).mean()),
            "maker_entry_share": float((t["entry"] == "limit").mean()),
            "avg_hold_min": float(((t["exit_time"] - t["entry_time"]).dt.total_seconds() / 60).mean())}
