"""Split a live account's P&L into market exposure, timing, selection and costs.

An account that is simply long the market makes money whenever the market
rises; that is beta, not skill. Per bar t (marks are taken at bar closes,
before that bar's trades), with ``m_t`` the equal-weight basket return of the
traded universe from close_{t-1} to close_t and ``x_t`` the signed exposure
the account held *through* the bar — positions after the bar-(t-1) trades,
valued at close_{t-1}, as a fraction of equity_{t-1}:

    total     = passive + timing + selection - costs
    passive   = mean(x) * sum(m_t * equity_{t-1})            holding the average exposure all along
    timing    = sum((x_t - mean(x)) * m_t * equity_{t-1})    varying exposure with the market
    selection = the rest, before costs                       which assets, and how much of each
    costs     = fees + spread/impact + funding that hit the window's marks

The exposure must be the start-of-bar one: exposure valued at close_t already
moves with m_t, which books part of every bar's market move as fake timing
(and the same amount, negated, as fake selection). ``beta`` is the OLS slope
of per-bar account returns on ``m_t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def basket_returns(close: pd.DataFrame) -> pd.Series:
    """Equal-weight basket return per bar (close_{t-1} → close_t), last price carried over gaps."""
    return close.ffill().pct_change(fill_method=None).mean(axis=1, skipna=True)


def held_units(fills: pd.DataFrame, bars: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Units held through each bar in ``bars``: every fill *decided* at a bar before it.

    ``fills`` has columns ``bar`` (decision bar; its trades fill right after that
    bar's mark), ``symbol`` and ``qty``.
    """
    out = np.zeros((len(bars), len(symbols)))
    if len(fills):
        q = (fills.groupby(["bar", "symbol"])["qty"].sum().unstack(fill_value=0.0)
             .reindex(columns=symbols, fill_value=0.0).sort_index())
        cum = q.cumsum().to_numpy()
        k = np.searchsorted(q.index.values, bars.values, side="left") - 1    # last decision bar < t
        out[k >= 0] = cum[k[k >= 0]]
    return pd.DataFrame(out, index=bars, columns=symbols)


def start_exposure(units: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    """Signed notional of ``units`` (held through bar t) valued at close_{t-1}."""
    prev = close.ffill().shift(1).reindex(index=units.index, columns=units.columns)
    return (units * prev.fillna(0.0)).sum(axis=1)


def bar_costs(fills: pd.DataFrame, funding: pd.DataFrame, marks: pd.DatetimeIndex) -> pd.DataFrame:
    """Costs per mark: fees and spread/impact of trades decided at bar b hit the first mark after b;
    funding settled at stamp s is charged at the first mark at or after s (before that mark is taken)."""
    out = pd.DataFrame(0.0, index=marks, columns=["fees", "spread", "funding"])
    vals = out.to_numpy()
    if len(fills):
        k = np.searchsorted(marks.values, fills["bar"].values, side="right")
        ok = k < len(marks)
        np.add.at(vals[:, 0], k[ok], fills["fee"].to_numpy(float)[ok])
        np.add.at(vals[:, 1], k[ok], fills["slippage_cost"].to_numpy(float)[ok])
    if len(funding):
        k = np.searchsorted(marks.values, funding["ts"].values, side="left")
        ok = k < len(marks)
        np.add.at(vals[:, 2], k[ok], -funding["amount"].to_numpy(float)[ok])
    return pd.DataFrame(vals, index=marks, columns=out.columns)


def attribute(equity: pd.Series, exposure: pd.Series, market: pd.Series,
              costs: pd.DataFrame | None = None) -> dict | None:
    """Attribute the P&L between the first usable mark of ``equity`` and its last mark.

    ``exposure`` is the start-of-bar signed notional per mark (see ``start_exposure``),
    ``market`` the basket return per mark and ``costs`` the per-mark cost frame
    (``bar_costs``). Returns None while fewer than two bars are usable.
    """
    df = pd.DataFrame({"eq": equity.astype(float), "x": exposure.reindex(equity.index),
                       "m": market.reindex(equity.index)})
    df["eq_prev"] = df["eq"].shift(1)
    usable = np.isfinite(df.to_numpy()).all(axis=1)
    if usable.sum() < 2:
        return None
    first = int(np.argmax(usable))                 # first bar with a market return and an exposure
    df = df.iloc[first:].copy()
    df[["x", "m"]] = df[["x", "m"]].fillna(0.0)    # later gaps: no market move, nothing earned
    start = equity.index[first - 1]
    c = (costs.reindex(df.index).fillna(0.0).sum() if costs is not None
         else pd.Series(0.0, index=["fees", "spread", "funding"]))
    w = df["x"] / df["eq_prev"]
    unit = df["m"] * df["eq_prev"]                 # P&L of a 1x long basket position per bar
    avg = float(w.mean())
    passive = avg * float(unit.sum())
    timing = float(((w - avg) * unit).sum())
    total = float(df["eq"].iloc[-1] - equity.loc[start])
    cost = float(c.sum())
    r = df["eq"] / df["eq_prev"] - 1.0
    var = float(df["m"].var())
    beta = float(np.cov(r, df["m"])[0, 1] / var) if var > 0 else None
    return {"window_start": str(start), "bars": int(len(df)), "pnl": total,
            "market_return": float(np.prod(1.0 + df["m"]) - 1.0), "avg_exposure": avg, "beta": beta,
            "passive_pnl": passive, "timing_pnl": timing, "selection_pnl": total - passive - timing + cost,
            "costs": cost, "fees": float(c["fees"]), "spread": float(c["spread"]), "funding": float(c["funding"])}
