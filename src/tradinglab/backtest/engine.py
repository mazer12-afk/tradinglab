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
                 risk: RiskConfig | None = None, start=None, end=None,
                 cost_bps: np.ndarray | None = None, use_kernel: bool = True) -> BacktestResult:
    """Backtest target weights. ``cost_bps`` optionally overrides costs per asset (bps per side)."""
    cols = list(target.columns)
    idx = target.index
    R = panel.returns().reindex(index=idx, columns=cols).to_numpy(dtype="float64")
    F = panel.funding.reindex(index=idx, columns=cols).fillna(0.0).to_numpy(dtype="float64")
    tgt = target.reindex(index=idx, columns=cols).fillna(0.0).to_numpy(dtype="float64")
    T, N = tgt.shape
    mask = np.ones(T, dtype=bool)
    if start is not None:
        mask &= idx >= utc(start)
    if end is not None:
        mask &= idx < utc(end)
    bps = (np.full(N, costs.total_bps) if cost_bps is None else np.asarray(cost_bps, dtype="float64")) * 1e-4
    finite_rows = np.isfinite(R).any(axis=1)
    last_valid = int(np.nonzero(finite_rows)[0][-1]) if finite_rows.any() else -1
    g = (1, risk.dd_derisk_start, risk.dd_halt, risk.dd_floor_scale, 20) if risk is not None else (0, 0.0, 1.0, 1.0, 0)
    kernel = _kernel_numba() if use_kernel else None
    fn = kernel if kernel is not None else _kernel_py
    net, gross_r, cost_r, fund_r, turn, held, scales = fn(tgt, R, F, mask, bps, float(band), last_valid, *g)

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


def _kernel_py(tgt, R, F, mask, bps, band, last_valid, use_guard, g_start, g_halt, g_floor, g_cool):
    """Reference implementation (also the numba kernel's source)."""
    T, N = tgt.shape
    w = np.zeros(N)
    equity = 1.0
    peak = np.nan
    halted = 0
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
        if t > last_valid:
            break  # future prices unknown from here on
        # Drawdown guard (same rules as portfolio.risk.DrawdownGuard).
        scale = 1.0
        if use_guard:
            if np.isnan(peak):
                peak = equity
            if halted > 0:
                halted -= 1
                if halted == 0:
                    peak = equity
                scale = 0.0
            else:
                peak = max(peak, equity)
                dd = 1.0 - equity / peak if peak > 0 else 0.0
                if dd >= g_halt:
                    halted = g_cool
                    scale = 0.0
                elif dd > g_start:
                    scale = 1.0 - (dd - g_start) / (g_halt - g_start) * (1.0 - g_floor)
        tv = 0.0
        c = 0.0
        gr = 0.0
        fnd = 0.0
        new_w = np.empty(N)
        for i in range(N):
            r = R[t, i]
            desired = 0.0 if np.isnan(r) else tgt[t, i] * scale
            if abs(desired - w[i]) > band or (desired == 0.0 and w[i] != 0.0):
                new_w[i] = desired
            else:
                new_w[i] = w[i]
            d = abs(new_w[i] - w[i])
            tv += d
            c += d * bps[i]
            if not np.isnan(r):
                gr += new_w[i] * r
            fnd += new_w[i] * F[t, i]
        n_ret = gr - fnd - c
        equity *= 1.0 + n_ret
        net[t] = n_ret
        gross_r[t] = gr
        cost_r[t] = c
        fund_r[t] = fnd
        turn[t] = tv
        scales[t] = scale
        denom = 1.0 + n_ret
        for i in range(N):
            held[t, i] = new_w[i]
            r = R[t, i]
            rr = 0.0 if np.isnan(r) else r
            w[i] = new_w[i] * (1.0 + rr) / denom if denom > 0 else 0.0
    return net, gross_r, cost_r, fund_r, turn, held, scales


_NUMBA_KERNEL = None


def _kernel_numba():
    """JIT-compiled kernel (None when numba is unavailable)."""
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is None:
        try:
            import numba

            _NUMBA_KERNEL = numba.njit(cache=True)(_kernel_py)
        except Exception:  # noqa: BLE001 - numba is optional
            _NUMBA_KERNEL = False
    return _NUMBA_KERNEL or None
