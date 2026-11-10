"""Books on the survivorship-free daily panel (research/proof/wide.py): perpetual books and spot-perp carry.

Perpetual book: weights W (days x coins) held over each day; P&L = W x return - W x funding - 6 bp per unit of
weight changed. Carry: an adapter that gives carry.py's frozen rules and book the same inputs they get from
CarryPanel, so R2/R4 run unchanged on every coin Binance ever listed on both spot and perpetuals.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import carry as C
from wide import Wide

COST_BPS = 6.0
PER_YEAR = 365


def perp_book(w: Wide, W: pd.DataFrame, cost_bps: float = COST_BPS, max_gross: float = 1.0) -> pd.DataFrame:
    W = W.reindex(index=w.days, columns=w.symbols).where(w.live, 0.0).fillna(0.0)
    W = W.div(np.maximum(W.abs().sum(axis=1) / max_gross, 1.0), axis=0)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    gross = (W * w.ret.fillna(0.0)).sum(axis=1)
    funding = (W * w.fund.fillna(0.0)).sum(axis=1)
    cost = turnover * cost_bps * 1e-4
    return pd.DataFrame({"gross": gross, "funding": -funding, "cost": -cost, "net": gross - funding - cost,
                         "turnover": turnover, "exposure": W.abs().sum(axis=1), "net_dollars": W.sum(axis=1),
                         "names": (W != 0).sum(axis=1)})


def market(w: Wide, top: int = 30, min_age: int = 30) -> pd.Series:
    """Equal-weight daily return of the ``top`` most traded coins (point in time) at least ``min_age`` days old."""
    ok = (w.liquidity_rank() <= top) & (w.age >= min_age) & w.live
    return w.ret.where(ok).mean(axis=1)


def stats(b: pd.DataFrame, mkt: pd.Series | None = None) -> dict:
    r = b["net"]
    eq = (1 + r).cumprod()
    sd = r.std()
    out = {"sharpe": round(float(r.mean() / sd * math.sqrt(PER_YEAR)), 3) if sd > 0 else 0.0,
           "total_return": round(float(eq.iloc[-1] - 1), 4), "cagr": round(float(eq.iloc[-1] ** (PER_YEAR / len(r)) - 1), 4),
           "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
           "by_year": {str(k): round(float(v), 4) for k, v in (1 + r).groupby(r.index.year).prod().sub(1).items()}}
    for k in ("cost", "funding", "basis"):
        if k in b:
            out[f"{k}_per_year"] = round(float(b[k].mean() * PER_YEAR), 4)
    if "turnover" in b:
        out["turnover_per_day"] = round(float(b["turnover"].mean()), 3)
    if mkt is not None:
        m = mkt.reindex(r.index).fillna(0.0)
        X = np.column_stack([np.ones(len(m)), m.to_numpy()])
        beta = np.linalg.lstsq(X, r.to_numpy(), rcond=None)[0]
        e = r.to_numpy() - X @ beta
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * e.var(ddof=2))
        out.update({"beta": round(float(beta[1]), 3), "alpha_per_year": round(float(beta[0] * PER_YEAR), 4), "alpha_t": round(float(beta[0] / se[0]), 2)})
    return out


class WideCarry:
    """What carry.py's rules and book read from a CarryPanel, built from the wide panel for ``symbols`` (perpetual
    names). A coin is tradable from 30 days after it trades on both markets until ``exit_before`` days before its
    perpetual's last print (delistings are announced days ahead; 0 = hold to the last daily close)."""

    def __init__(self, w: Wide, symbols: list[str] | None = None, start: str | None = None, end: str | None = None,
                 exit_before: int = 2, min_days: int = C.MIN_DAYS, top: int | None = None,
                 min_spot_volume: float = 0.0, max_gap: float | None = None):
        """``top``: only coins among the ``top`` most traded perpetuals (30-day median volume, point in time);
        ``min_spot_volume``: 30-day median spot volume in USDT; ``max_gap``: largest |log(spot / perp)| at the
        day's open (a wider gap means a halted market or a different asset under the same ticker)."""
        syms = [s for s in (symbols or w.symbols) if s in w.symbols and w.spot_all[s].notna().any()]
        days = w.days
        if start:
            days = days[days >= pd.Timestamp(start, tz="UTC")]
        if end:
            days = days[days < pd.Timestamp(end, tz="UTC")]
        self.grid, self.symbols = days, syms
        self.spot_ret = w.spot_ret.loc[days, syms]
        self.perp_ret = w.ret.loc[days, syms]
        self.fund_next = w.fund.loc[days, syms].fillna(0.0)
        self._fund_all = w.fund_all[syms].fillna(0.0)
        first = pd.Series({s: max(w.spot_all[s].first_valid_index(), w.first[s]) for s in syms})
        last = w.last[syms]
        still = last >= w.px_all.index[-1] - pd.Timedelta(days=1)             # trading at the panel's end
        self.live = pd.DataFrame({s: (days >= first[s] + pd.Timedelta(days=min_days)) &
                                     (still[s] | (days <= last[s] - pd.Timedelta(days=exit_before + 1)))
                                  for s in syms}, index=days)
        self.live &= self.spot_ret.notna() & self.perp_ret.notna()
        if top is not None:
            self.live &= (w.liquidity_rank().loc[days, syms] <= top)
        if min_spot_volume > 0:
            self.live &= (w.spot_qv_all[syms].rolling(30, min_periods=10).median().reindex(days) >= min_spot_volume)
        if max_gap is not None:
            gap = np.log(w.spot_all[syms].mul(w.scale[syms], axis=1) / w.px_all[syms]).abs().reindex(days)
            self.live &= (gap <= max_gap)

    def trailing_funding(self, days: int) -> pd.DataFrame:
        """Average daily funding over the ``days`` days before each day (settlements up to and including 00:00)."""
        return self._fund_all.rolling(days, min_periods=1).sum().shift(1).reindex(self.grid) / days
