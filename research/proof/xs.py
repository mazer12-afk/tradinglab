"""Cross-sectional perpetual-futures books on hourly data, for any coins and period (the proof tests' engine).

A rule maps a panel of per-coin features at each 4-hour origin to target weights; the book holds them for 4 hours
(rebalancing every 4 hours, gross exposure at most 1), pays COST_BPS per unit of weight changed and the funding of
every settlement held through. Coins join once they have MIN_DAYS of history.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tradinglab.config import load_config
from tradinglab.data.store import MarketData

COST_BPS = 6.0              # 5 bp taker fee + 1 bp slippage per unit traded
MIN_DAYS = 30
PER_YEAR = 6 * 365
CFG = load_config()
TWELVE = CFG.intraday.symbols
OTHERS = [s for s in CFG.universe.symbols if s not in TWELVE]


class Panel:
    """Hourly closes and funding for a set of coins, and the 4-hourly grid the books trade on."""

    def __init__(self, symbols: list[str], start: str, end: str):
        md = MarketData(CFG.paths.data_dir)
        s0 = pd.Timestamp(start, tz="UTC")
        self.start, self.end = s0, pd.Timestamp(end, tz="UTC")
        load_from = (s0 - pd.Timedelta(days=60)).tz_localize(None)
        close, vol, fund = {}, {}, {}
        for s in symbols:
            try:
                k = md.klines(s, "1h", load_from, market="um")
            except Exception:                      # noqa: BLE001 - a coin not listed yet
                continue
            k = k[k.index < self.end + pd.Timedelta(hours=8)]
            if len(k) < 24 * MIN_DAYS:
                continue
            close[s], vol[s] = k["close"].astype(float), k["quote_volume"].astype(float)
            r = md.funding_8h(s, load_from)
            idx = pd.DatetimeIndex(r.index)
            r.index = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")).round("h")
            fund[s] = r[~r.index.duplicated(keep="last")]
        self.close = pd.DataFrame(close).sort_index()
        self.close = self.close.reindex(pd.date_range(self.close.index[0], self.close.index[-1], freq="1h"))
        self.qvol = pd.DataFrame(vol).reindex(self.close.index)
        self.symbols = list(self.close.columns)
        self.lc = np.log(self.close)
        self.hr = self.lc.diff()
        self.grid = pd.date_range(s0, self.end, freq="4h", inclusive="left")
        prev = self.grid - pd.Timedelta(hours=1)                          # last closed candle before each origin
        self.price = self.lc.reindex(prev).set_axis(self.grid)
        nxt = self.lc.reindex(self.grid + pd.Timedelta(hours=3)).set_axis(self.grid)
        self.r4 = np.exp(nxt - self.price) - 1                            # the next 4 hours' simple return
        first = self.close.apply(lambda c: c.first_valid_index())
        self.live = pd.DataFrame({s: self.grid >= first[s] + pd.Timedelta(days=MIN_DAYS) for s in self.symbols}, index=self.grid)
        self.live &= self.price.notna() & self.r4.notna()
        F = {}
        for s in self.symbols:
            fr = fund.get(s, pd.Series(dtype=float))
            ft, fc = fr.index.to_numpy(), np.concatenate([[0.0], np.cumsum(fr.to_numpy())])
            a = np.searchsorted(ft, self.grid.to_numpy(), side="right")
            b = np.searchsorted(ft, (self.grid + pd.Timedelta(hours=4)).to_numpy(), side="right")
            F[s] = fc[b] - fc[a]
        self.fund4 = pd.DataFrame(F, index=self.grid)

    def hourly_vol(self, hours: int) -> pd.DataFrame:
        """Std of hourly log returns over the ``hours`` candles before each origin."""
        v = self.hr.rolling(hours, min_periods=int(hours * 0.8)).std()
        return v.reindex(self.grid - pd.Timedelta(hours=1)).set_axis(self.grid)

    def past_return(self, hours: int) -> pd.DataFrame:
        return self.price - self.lc.reindex(self.grid - pd.Timedelta(hours=1 + hours)).set_axis(self.grid)

    def market(self) -> pd.Series:
        """Equal-weight 4-hour return of the live coins (the market the books are measured against)."""
        return self.r4.where(self.live).mean(axis=1)


def zscore_weights(sig: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    s = sig.where(live)
    z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1) + 1e-12, axis=0)
    return z.div(z.abs().sum(axis=1), axis=0).fillna(0.0)


def ease(target: pd.DataFrame, step: float) -> pd.DataFrame:
    out = np.zeros(target.shape)
    cur = np.zeros(target.shape[1])
    for i, row in enumerate(target.to_numpy()):
        cur = cur + step * (row - cur)
        out[i] = cur
    return pd.DataFrame(out, index=target.index, columns=target.columns)


def book(p: Panel, W: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Per-period gross P&L, funding, costs and net return for weights W (gross capped at 1, dead coins zeroed)."""
    W = W.where(p.live, 0.0).fillna(0.0)
    W = W.div(np.maximum(W.abs().sum(axis=1), 1.0), axis=0)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    gross = (W * p.r4.fillna(0.0)).sum(axis=1)
    funding = (W * p.fund4.fillna(0.0)).sum(axis=1)
    cost = turnover * cost_bps * 1e-4
    return pd.DataFrame({"gross": gross, "funding": -funding, "cost": -cost, "net": gross - funding - cost,
                         "turnover": turnover, "exposure": W.abs().sum(axis=1), "net_dollars": W.sum(axis=1)})


def stats(b: pd.DataFrame, mkt: pd.Series | None = None) -> dict:
    r = b["net"]
    eq = (1 + r).cumprod()
    sd = r.std()
    out = {"sharpe": round(float(r.mean() / sd * math.sqrt(PER_YEAR)), 3) if sd > 0 else 0.0,
           "total_return": round(float(eq.iloc[-1] - 1), 4),
           "cagr": round(float(eq.iloc[-1] ** (PER_YEAR / len(r)) - 1), 4),
           "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
           "turnover_per_day": round(float(b["turnover"].mean() * 6), 3),
           "cost_per_year": round(float(-b["cost"].mean() * PER_YEAR), 4),
           "funding_per_year": round(float(b["funding"].mean() * PER_YEAR), 4),
           "by_year": {str(k): round(float(v), 4) for k, v in (1 + r).groupby(r.index.year).prod().sub(1).items()}}
    if mkt is not None:
        m = mkt.reindex(r.index).fillna(0.0)
        X = np.column_stack([np.ones(len(m)), m.to_numpy()])
        beta, res = np.linalg.lstsq(X, r.to_numpy(), rcond=None)[:2]
        e = r.to_numpy() - X @ beta
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * e.var(ddof=2))
        out.update({"beta": round(float(beta[1]), 3), "alpha_per_year": round(float(beta[0] * PER_YEAR), 4),
                    "alpha_t": round(float(beta[0] / se[0]), 2)})
    return out
