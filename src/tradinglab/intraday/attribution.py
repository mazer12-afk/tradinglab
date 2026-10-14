"""Split a live account's P&L into market exposure and skill.

An account that is simply long the market makes money whenever the market
rises; that is beta, not skill. Per bar t, with ``m_t`` the equal-weight basket
return of the traded universe and ``net_t`` the account's net exposure
(signed notional / equity) held through the bar:

    market P&L_t = net_t * m_t * equity_{t-1}

and over the window

    total     = passive + timing + selection - costs
    passive   = mean(net) * sum(m_t * equity_{t-1})        holding the average exposure all along
    timing    = sum((net_t - mean(net)) * m_t * equity_{t-1})   varying exposure with the market
    selection = the rest, before fees and funding        which assets, and how much of each

``beta`` is the OLS slope of per-bar account returns on ``m_t`` (a static
summary; for accounts whose exposure swings it mixes passive and timing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def basket_returns(close: pd.DataFrame) -> pd.Series:
    """Equal-weight basket return per bar (close_{t-1} → close_t), last price carried over gaps."""
    return close.ffill().pct_change(fill_method=None).mean(axis=1, skipna=True)


def attribute(equity: pd.DataFrame, close: pd.DataFrame, costs: float = 0.0) -> dict | None:
    """Attribute P&L between the first and last mark of ``equity`` (columns ``equity``, ``net``).

    ``costs`` are the fees and funding paid between those marks (positive = paid),
    so ``selection`` measures asset choice before trading costs. Returns None
    while fewer than two marks have market data.
    """
    m = basket_returns(close).reindex(equity.index)
    df = pd.DataFrame({"eq": equity["equity"].astype(float), "net": equity["net"].astype(float), "m": m})
    df["eq_prev"] = df["eq"].shift(1)
    df = df.iloc[1:]
    df = df[np.isfinite(df[["eq", "net", "m", "eq_prev"]]).all(axis=1)]
    if len(df) < 2:
        return None
    start = equity.index[equity.index.get_loc(df.index[0]) - 1]
    unit = df["m"] * df["eq_prev"]                   # P&L of a 1x long basket position per bar
    avg_net = float(df["net"].mean())
    passive = avg_net * float(unit.sum())
    timing = float(((df["net"] - avg_net) * unit).sum())
    total = float(df["eq"].iloc[-1] - equity.loc[start, "equity"])
    r = df["eq"] / df["eq_prev"] - 1.0
    var = float(df["m"].var())
    beta = float(np.cov(r, df["m"])[0, 1] / var) if var > 0 else float("nan")
    return {"window_start": str(start), "bars": int(len(df)), "pnl": total,
            "market_return": float(np.prod(1.0 + df["m"]) - 1.0), "avg_net": avg_net, "beta": beta,
            "passive_pnl": passive, "timing_pnl": timing, "selection_pnl": total - passive - timing + costs,
            "costs": float(costs)}
