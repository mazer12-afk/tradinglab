"""Event-accurate portfolio backtester.

Timeline for the bar labelled ``t`` (its close time):

* target weights are computed from data with index <= t;
* trades execute at ``exec_price[t]`` (t + exec lag) paying fee + slippage on
  the traded notional; trades smaller than the rebalance band are skipped;
* the resulting positions earn ``exec_price[t+1] / exec_price[t] - 1`` and pay
  perpetual funding accrued over the holding interval;
* weights then drift with prices until the next rebalance.

The live paper broker implements the same accounting on unit positions; the
parity test in ``tests/test_parity.py`` checks both agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradinglab.backtest.metrics import summarize
from tradinglab.config import CostConfig, RiskConfig, utc
from tradinglab.data.store import Panel
from tradinglab.portfolio.risk import DrawdownGuard


@dataclass
class BacktestResult:
    returns: pd.Series      # net return realised over (exec_t, exec_t+1]
    gross: pd.Series
    costs: pd.Series
    funding: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame   # post-trade weights held from t
    exposure_scale: pd.Series

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    def metrics(self, periods: float) -> dict:
        m = summarize(self.returns, periods, self.turnover, self.costs)
        if m:
            m["funding_drag_ann"] = float(self.funding.reindex(self.returns.index).mean() * periods)
            m["avg_gross"] = float(self.weights.abs().sum(axis=1).reindex(self.returns.index).mean())
            m["avg_net"] = float(self.weights.sum(axis=1).reindex(self.returns.index).mean())
        return m


def run_backtest(target: pd.DataFrame, panel: Panel, costs: CostConfig, band: float = 0.0,
                 risk: RiskConfig | None = None, start=None, end=None) -> BacktestResult:
    cols = list(target.columns)
    idx = target.index
    R = panel.returns().reindex(index=idx, columns=cols).to_numpy()
    F = panel.funding.reindex(index=idx, columns=cols).fillna(0.0).to_numpy()
    tgt = target.reindex(index=idx, columns=cols).fillna(0.0).to_numpy()
    T, N = tgt.shape

    mask = np.ones(T, dtype=bool)
    if start is not None:
        mask &= idx >= utc(start)
    if end is not None:
        mask &= idx < utc(end)

    bps = costs.total_bps * 1e-4
    guard = DrawdownGuard(risk) if risk is not None else None
    w = np.zeros(N)
    equity = 1.0
    net = np.full(T, np.nan)
    gross_r = np.full(T, np.nan)
    cost_r = np.full(T, np.nan)
    fund_r = np.full(T, np.nan)
    turn = np.full(T, np.nan)
    held = np.zeros((T, N))
    scales = np.full(T, np.nan)

    for t in range(T):
        if not mask[t]:
            continue
        r = R[t]
        if np.all(np.isnan(r)):
            # Future (or missing) prices: cannot realise this bar.
            if t == T - 1 or not np.isfinite(R[t:]).any():
                break
        scale = guard.update(equity) if guard is not None else 1.0
        desired = tgt[t] * scale
        # Assets with no next price can't be held (delisting / data gap): exit.
        unknown = np.isnan(r)
        desired = np.where(unknown, 0.0, desired)
        delta = desired - w
        trade = np.abs(delta) > band
        trade |= (desired == 0.0) & (w != 0.0)
        new_w = np.where(trade, desired, w)
        tv = float(np.abs(new_w - w).sum())
        c = tv * bps
        rr = np.nan_to_num(r, nan=0.0)
        g = float(new_w @ rr)
        fnd = float(new_w @ F[t])
        n_ret = g - fnd - c
        equity *= 1.0 + n_ret
        net[t], gross_r[t], cost_r[t], fund_r[t], turn[t] = n_ret, g, c, fnd, tv
        held[t] = new_w
        scales[t] = scale
        # Drift weights with realised returns.
        denom = 1.0 + n_ret
        w = new_w * (1.0 + rr) / denom if denom > 0 else np.zeros(N)

    s = lambda a: pd.Series(a, index=idx).dropna()  # noqa: E731
    ret = s(net)
    return BacktestResult(
        returns=ret,
        gross=s(gross_r),
        costs=s(cost_r),
        funding=s(fund_r),
        turnover=s(turn),
        weights=pd.DataFrame(held, index=idx, columns=cols).loc[ret.index],
        exposure_scale=s(scales),
    )
