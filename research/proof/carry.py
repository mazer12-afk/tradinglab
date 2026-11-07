"""Funding carry ("cash and carry"): long spot and short the USD-M perpetual of the same coin, equal notional, so
price moves cancel and the short collects the funding rate. Positions are reviewed once a day (00:00 UTC).

Accounting per unit of notional: spot return - perpetual return (the basis change) + funding received by the
short at every settlement, minus costs on every change of position (spot 10 bp fee + 1 bp slippage, perpetual 5 bp
+ 1 bp), and on the daily trimming of both legs back to their share of capital as the price moves (charged
as if each leg were rebalanced by the day's price change). Capital = notional x CAPITAL_PER_NOTIONAL: the spot leg paid in full plus 50 % margin on the short, so a
spike of up to ~50 % cannot liquidate it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tradinglab.config import load_config
from tradinglab.data.store import MarketData

SPOT_BPS, PERP_BPS = 11.0, 6.0
CAPITAL_PER_NOTIONAL = 1.5
MIN_DAYS = 30
CFG = load_config()
TWELVE = CFG.intraday.symbols
OTHERS = [s for s in CFG.universe.symbols if s not in TWELVE]


class ArchiveSource:
    """Hourly spot and perpetual closes and funding settlements from Binance's public archive."""

    def __init__(self):
        self.md = MarketData(CFG.paths.data_dir)

    def spot(self, s: str, start: pd.Timestamp) -> pd.Series:
        return self.md.klines(s, "1h", start.tz_localize(None), market="spot")["close"].astype(float)

    def perp(self, s: str, start: pd.Timestamp) -> pd.Series:
        return self.md.klines(s, "1h", start.tz_localize(None), market="um")["close"].astype(float)

    def funding(self, s: str, start: pd.Timestamp) -> pd.Series:
        return self.md.funding_8h(s, start.tz_localize(None))


class CarryPanel:
    """Daily grid (00:00 UTC) with each coin's spot and perpetual returns over the next day and the funding paid in
    it, and the funding history up to each day (for the rules). ``source`` supplies the data (default: archive)."""

    def __init__(self, symbols: list[str], start: str, end: str, source=None):
        src = source or ArchiveSource()
        s0, s1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        lf = s0 - pd.Timedelta(days=60)
        self.grid = pd.date_range(s0, s1, freq="D", inclusive="left")
        sp, pp, fr = {}, {}, {}
        for s in symbols:
            try:
                a, b = src.spot(s, lf), src.perp(s, lf)
            except Exception:                      # noqa: BLE001 - not listed on one of the two markets
                continue
            if a.empty or b.empty:
                continue
            r = src.funding(s, lf)
            idx = pd.DatetimeIndex(r.index)
            r.index = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")).round("h")
            sp[s], pp[s], fr[s] = a, b, r[~r.index.duplicated(keep="last")]
        self.symbols = list(sp)
        # price at each day's open = close of the 23:00 candle the day before
        at = lambda c: c.reindex(self.grid - pd.Timedelta(hours=1)).set_axis(self.grid)      # noqa: E731
        S, P = pd.DataFrame({s: at(sp[s]) for s in self.symbols}), pd.DataFrame({s: at(pp[s]) for s in self.symbols})
        self.spot_ret = S.shift(-1) / S - 1                                  # over the next day
        self.perp_ret = P.shift(-1) / P - 1
        F, cnt = {}, {}
        for s in self.symbols:
            f = fr[s]
            ft, fc = f.index.to_numpy(), np.concatenate([[0.0], np.cumsum(f.to_numpy())])
            a = np.searchsorted(ft, self.grid.to_numpy(), side="right")
            b = np.searchsorted(ft, (self.grid + pd.Timedelta(days=1)).to_numpy(), side="right")
            F[s] = fc[b] - fc[a]                                             # funding over (day, day + 1]
            cnt[s] = b - a
        self.fund_next = pd.DataFrame(F, index=self.grid)
        self.fund_hist = {s: fr[s] for s in self.symbols}
        first = pd.Series({s: max(sp[s].first_valid_index(), pp[s].first_valid_index()) for s in self.symbols})
        self.live = pd.DataFrame({s: self.grid >= first[s] + pd.Timedelta(days=MIN_DAYS) for s in self.symbols}, index=self.grid)
        self.live &= self.spot_ret.notna() & self.perp_ret.notna()

    def trailing_funding(self, days: int) -> pd.DataFrame:
        """Sum of the funding settled over the ``days`` days before each day's open (per day)."""
        out = {}
        for s in self.symbols:
            f = self.fund_hist[s]
            ft, fc = f.index.to_numpy(), np.concatenate([[0.0], np.cumsum(f.to_numpy())])
            b = np.searchsorted(ft, self.grid.to_numpy(), side="right")
            a = np.searchsorted(ft, (self.grid - pd.Timedelta(days=days)).to_numpy(), side="right")
            out[s] = (fc[b] - fc[a]) / days
        return pd.DataFrame(out, index=self.grid)


def book(p: CarryPanel, N: pd.DataFrame) -> pd.DataFrame:
    """Daily P&L on capital for notional weights N (fraction of the maximum notional per coin; row sum <= 1)."""
    N = N.where(p.live, 0.0).fillna(0.0)
    N = N.div(np.maximum(N.sum(axis=1), 1.0), axis=0)
    change = N.diff().abs()
    change.iloc[0] = N.iloc[0].abs()
    basis = (N * (p.spot_ret - p.perp_ret).fillna(0.0)).sum(axis=1)
    funding = (N * p.fund_next.fillna(0.0)).sum(axis=1)
    # entering/leaving positions, plus trimming both legs back to their share of capital as the price moves
    drift = (N * p.spot_ret.abs().fillna(0.0))
    cost = ((change + drift) * (SPOT_BPS + PERP_BPS) * 1e-4).sum(axis=1)
    k = CAPITAL_PER_NOTIONAL
    return pd.DataFrame({"basis": basis / k, "funding": funding / k, "cost": -cost / k,
                         "net": (basis + funding - cost) / k, "notional": N.sum(axis=1), "coins": (N > 0).sum(axis=1)})


def stats(b: pd.DataFrame, mkt: pd.Series | None = None) -> dict:
    r = b["net"]
    eq = (1 + r).cumprod()
    sd = r.std()
    out = {"sharpe": round(float(r.mean() / sd * math.sqrt(365)), 3) if sd > 0 else 0.0,
           "total_return": round(float(eq.iloc[-1] - 1), 4), "cagr": round(float(eq.iloc[-1] ** (365 / len(r)) - 1), 4),
           "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
           "funding_per_year": round(float(b["funding"].mean() * 365), 4), "basis_per_year": round(float(b["basis"].mean() * 365), 4),
           "cost_per_year": round(float(b["cost"].mean() * 365), 4), "avg_notional": round(float(b["notional"].mean()), 3),
           "by_year": {str(k): round(float(v), 4) for k, v in (1 + r).groupby(r.index.year).prod().sub(1).items()}}
    if mkt is not None:
        m = mkt.reindex(r.index).fillna(0.0)
        X = np.column_stack([np.ones(len(m)), m.to_numpy()])
        beta = np.linalg.lstsq(X, r.to_numpy(), rcond=None)[0]
        e = r.to_numpy() - X @ beta
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * e.var(ddof=2))
        out.update({"beta": round(float(beta[1]), 3), "alpha_per_year": round(float(beta[0] * 365), 4), "alpha_t": round(float(beta[0] / se[0]), 2)})
    return out


def hysteresis(sig: pd.DataFrame, live: pd.DataFrame, enter: float, leave: float) -> pd.DataFrame:
    """1 while a coin's signal has risen above ``enter`` and not yet fallen below ``leave``, else 0 (causal)."""
    S, L = sig.to_numpy(), live.to_numpy()
    on = np.zeros(S.shape)
    cur = np.zeros(S.shape[1], dtype=bool)
    for i in range(len(S)):
        x = S[i]
        cur = np.where(np.isnan(x) | ~L[i], False, np.where(cur, x > leave, x > enter))
        on[i] = cur
    return pd.DataFrame(on, index=sig.index, columns=sig.columns)


def funding_carry(p: CarryPanel, lookback_days: int = 7, enter_apr: float = 0.10, leave_apr: float = 0.0,
                  max_share: float = 0.25) -> pd.DataFrame:
    """Hold spot-long / perp-short on every coin whose average funding over the last ``lookback_days`` exceeds
    ``enter_apr`` a year, until it falls below ``leave_apr``; equal notional across held coins, at most
    ``max_share`` of the book each (the rest idle)."""
    sig = p.trailing_funding(lookback_days) * 365
    on = hysteresis(sig, p.live, enter_apr, leave_apr)
    k = on.sum(axis=1)
    w = on.div(np.maximum(k, 1.0 / max_share), axis=0)
    return w
