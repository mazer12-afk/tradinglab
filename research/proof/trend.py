"""Time-series trend following on USD-M perpetuals, daily (the proof tests' engine for the trend family).

Each day at 00:00 UTC every live coin gets a position in the direction of its trend, sized so each coin carries the
same volatility; the book's gross exposure is capped at 1 (no leverage). Costs: 6 bp per unit traded; funding paid
or received at every settlement held through.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tradinglab.config import load_config
from tradinglab.data.store import MarketData

COST_BPS = 6.0
MIN_DAYS = 60
CFG = load_config()
TWELVE = CFG.intraday.symbols
OTHERS = [s for s in CFG.universe.symbols if s not in TWELVE]


class DailyPanel:
    def __init__(self, symbols: list[str], start: str, end: str):
        md = MarketData(CFG.paths.data_dir)
        s0, s1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        lf = (s0 - pd.Timedelta(days=400)).tz_localize(None)
        self.grid = pd.date_range(s0, s1, freq="D", inclusive="left")
        px, fr = {}, {}
        for s in symbols:
            try:
                k = md.klines(s, "1h", lf, market="um")["close"].astype(float)
            except Exception:                      # noqa: BLE001
                continue
            if k.empty:
                continue
            r = md.funding_8h(s, lf)
            idx = pd.DatetimeIndex(r.index)
            r.index = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")).round("h")
            px[s], fr[s] = k, r[~r.index.duplicated(keep="last")]
        self.symbols = list(px)
        full = pd.date_range(min(v.index[0] for v in px.values()).floor("D"), s1, freq="D")
        # daily price at 00:00 = close of the 23:00 candle the day before
        self.px = pd.DataFrame({s: px[s].reindex(full - pd.Timedelta(hours=1)).to_numpy() for s in self.symbols}, index=full)
        self.ret = self.px.shift(-1) / self.px - 1                           # next day's return
        F = {}
        for s in self.symbols:
            f = fr[s]
            ft, fc = f.index.to_numpy(), np.concatenate([[0.0], np.cumsum(f.to_numpy())])
            a = np.searchsorted(ft, full.to_numpy(), side="right")
            b = np.searchsorted(ft, (full + pd.Timedelta(days=1)).to_numpy(), side="right")
            F[s] = fc[b] - fc[a]
        self.fund = pd.DataFrame(F, index=full)
        first = self.px.apply(lambda c: c.first_valid_index())
        self.live = pd.DataFrame({s: full >= first[s] + pd.Timedelta(days=MIN_DAYS) for s in self.symbols}, index=full)
        self.live &= self.px.notna() & self.ret.notna()


def trend_weights(p: DailyPanel, lookback: int = 60, vol_days: int = 30, kind: str = "sign", smooth: int = 1) -> pd.DataFrame:
    """Long coins whose ``lookback``-day return is positive, short the negative (``kind`` "sign"), or weight by the
    return in units of its volatility (``kind`` "scaled", capped at +-1); each coin's position is inversely
    proportional to its ``vol_days`` volatility so all carry the same risk; signals averaged over ``smooth`` days."""
    lp = np.log(p.px)
    past = lp - lp.shift(lookback)
    vol = lp.diff().rolling(vol_days, min_periods=int(vol_days * 0.8)).std()
    if kind == "sign":
        sig = np.sign(past)
    else:
        sig = (past / (vol * math.sqrt(lookback))).clip(-1, 1)
    if smooth > 1:
        sig = sig.ewm(span=smooth).mean()
    w = (sig / vol).where(p.live)
    return w.div(w.abs().sum(axis=1), axis=0).fillna(0.0)          # equal risk per coin, gross 1


def book(p: DailyPanel, W: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    W = W.reindex(p.grid).where(p.live.reindex(p.grid), 0.0).fillna(0.0)
    W = W.div(np.maximum(W.abs().sum(axis=1), 1.0), axis=0)
    R, F = p.ret.reindex(p.grid).fillna(0.0), p.fund.reindex(p.grid).fillna(0.0)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    gross = (W * R).sum(axis=1)
    funding = (W * F).sum(axis=1)
    cost = turnover * cost_bps * 1e-4
    return pd.DataFrame({"gross": gross, "funding": -funding, "cost": -cost, "net": gross - funding - cost,
                         "turnover": turnover, "net_dollars": W.sum(axis=1)})


def market(p: DailyPanel) -> pd.Series:
    return p.ret.where(p.live).mean(axis=1).reindex(p.grid)


def stats(b: pd.DataFrame, mkt: pd.Series | None = None) -> dict:
    r = b["net"]
    eq = (1 + r).cumprod()
    sd = r.std()
    out = {"sharpe": round(float(r.mean() / sd * math.sqrt(365)), 3) if sd > 0 else 0.0,
           "total_return": round(float(eq.iloc[-1] - 1), 4), "cagr": round(float(eq.iloc[-1] ** (365 / len(r)) - 1), 4),
           "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
           "turnover_per_day": round(float(b["turnover"].mean()), 3), "cost_per_year": round(float(b["cost"].mean() * 365), 4),
           "funding_per_year": round(float(b["funding"].mean() * 365), 4), "avg_net_dollars": round(float(b["net_dollars"].mean()), 3),
           "by_year": {str(k): round(float(v), 4) for k, v in (1 + r).groupby(r.index.year).prod().sub(1).items()}}
    if mkt is not None:
        m = mkt.reindex(r.index).fillna(0.0)
        X = np.column_stack([np.ones(len(m)), m.to_numpy()])
        beta = np.linalg.lstsq(X, r.to_numpy(), rcond=None)[0]
        e = r.to_numpy() - X @ beta
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * e.var(ddof=2))
        out.update({"beta": round(float(beta[1]), 3), "alpha_per_year": round(float(beta[0] * 365), 4), "alpha_t": round(float(beta[0] / se[0]), 2)})
    return out


def vol_target(p: DailyPanel, W: pd.DataFrame, target: float = 0.20, days: int = 30) -> pd.DataFrame:
    """Scale the whole book so its recent volatility (the unscaled book's daily returns over the last ``days``
    days, known before each day) is ``target`` a year; never above gross 1."""
    raw = (W.reindex(p.px.index).fillna(0.0) * p.ret.fillna(0.0)).sum(axis=1)       # unscaled book, day by day
    est = raw.shift(1).rolling(days, min_periods=int(days * 0.8)).std() * math.sqrt(365)
    scale = (target / est).clip(upper=1.0 / np.maximum(W.abs().sum(axis=1).reindex(est.index).replace(0, np.nan), 1e-9)).fillna(0.0)
    return W.mul(scale.reindex(W.index), axis=0)
