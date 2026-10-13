"""Walk-forward research for intraday forecasters and the trading rule built on them.

Forecast → trade:
1. every bar each model forecasts the vol-normalised return over the next
   ``H`` bars;
2. the position is the average of the last ``H`` forecasts (overlapping
   portfolios, as in Jegadeesh–Titman: each forecast is "held" for its horizon),
   normalised by its own trailing dispersion and clipped to [-1, 1];
3. the shared sizing layer converts positions into vol-targeted weights;
4. trades fill at the next bar's open, paying taker fee + per-asset slippage and
   real funding.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradinglab.backtest.engine import BacktestResult, run_backtest
from tradinglab.backtest.metrics import summarize
from tradinglab.config import CostConfig, IntradayConfig, PortfolioConfig, bars_per_year, utc
from tradinglab.intraday.data import IntradayData
from tradinglab.intraday.features import FeatureSet
from tradinglab.intraday.models import FORECASTERS
from tradinglab.portfolio.sizing import size_weights

log = logging.getLogger(__name__)


def month_starts(index: pd.DatetimeIndex, first: str, step_months: int = 1) -> list[pd.Timestamp]:
    t, last, out = utc(first), index[-1], []
    while t <= last:
        out.append(t)
        t = t + pd.DateOffset(months=step_months)
    return out


@dataclass
class WalkForward:
    predictions: pd.DataFrame
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    folds: list[dict] = field(default_factory=list)


def train_rows(fs: FeatureSet, cfg: IntradayConfig, end: pd.Timestamp):
    """Training / validation row indices for a model used from ``end`` onwards."""
    idx = fs.index
    t_end = int(np.searchsorted(idx, end)) - cfg.horizon_bars - 1      # labels must be known before `end`
    t_start = int(np.searchsorted(idx, end - pd.Timedelta(days=cfg.train_days)))
    t_val = int(np.searchsorted(idx, end - pd.Timedelta(days=cfg.valid_days)))
    gap = cfg.horizon_bars
    tr = np.arange(max(t_start, 0), max(t_val - gap, 0), cfg.row_stride)
    va = np.arange(t_val, max(t_end, t_val), cfg.row_stride)
    return tr, va


def fit_forecaster(kind: str, fs: FeatureSet, cfg: IntradayConfig, end: pd.Timestamp, threads: int = 4,
                   seed: int = 0):
    tr, va = train_rows(fs, cfg, end)
    X, y, _, a = fs.rows(tr)
    Xv, yv, _, av = fs.rows(va)
    model = FORECASTERS[kind](threads=threads, seed=seed) if kind == "lgbm" else FORECASTERS[kind]()
    model.fit(X, y, Xv, yv, asset=a, asset_v=av)
    return model, {"train_rows": int(len(y)), "valid_rows": int(len(yv))}


def predict_range(model, fs: FeatureSet, start, end=None) -> pd.DataFrame:
    idx = fs.index
    lo = int(np.searchsorted(idx, utc(start)))
    hi = len(idx) if end is None else int(np.searchsorted(idx, utc(end)))
    t_idx = np.arange(lo, hi)
    X, _, tt, aa = fs.rows(t_idx, need_y=False)
    out = np.full((len(idx), len(fs.symbols)), np.nan)
    if len(X):
        out[tt, aa] = model.predict(X, asset=aa)
    return pd.DataFrame(out[lo:hi], index=idx[lo:hi], columns=fs.symbols)


def walk_forward(kind: str, fs: FeatureSet, cfg: IntradayConfig, threads: int = 4) -> WalkForward:
    starts = month_starts(fs.index, cfg.first_test, cfg.retrain_months)
    preds, folds, imps = [], [], []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else None
        t0 = time.time()
        model, info = fit_forecaster(kind, fs, cfg, s, threads)
        p = predict_range(model, fs, s, e)
        preds.append(p)
        if hasattr(model, "feature_importance"):
            imps.append(pd.Series(model.feature_importance()[: len(fs.names)], index=fs.names, name=str(s.date())))
        info.update({"start": str(s.date()), "seconds": round(time.time() - t0, 1),
                     "iterations": getattr(model, "best_iteration", None)})
        folds.append(info)
        log.info("%s fold %s: %s", kind, s.date(), info)
    imp = pd.DataFrame(imps) if imps else pd.DataFrame()
    return WalkForward(pd.concat(preds).sort_index(), imp, folds)


# --------------------------------------------------------------- trading rule
def forecast_to_signal(pred: pd.DataFrame, horizon: int, norm_window: int = 288 * 30) -> pd.DataFrame:
    """Average the last ``horizon`` forecasts, scale by trailing dispersion, clip to [-1, 1]."""
    s = pred.rolling(horizon, min_periods=1).mean()
    scale = s.abs().rolling(norm_window, min_periods=288).mean()
    return (s / (2.0 * scale)).clip(-1.0, 1.0)


def intraday_portfolio_config(cfg: IntradayConfig) -> PortfolioConfig:
    return PortfolioConfig(initial_capital=cfg.initial_capital, target_vol=cfg.target_vol,
                           asset_vol_target=0.6, vol_lookback=cfg.vol_span, max_gross=cfg.max_gross,
                           max_weight=cfg.max_weight, rebalance_band=cfg.rebalance_band,
                           min_trade_notional=cfg.min_trade_notional, signal_span=1)


def signal_to_weights(signal: pd.DataFrame, d: IntradayData, cfg: IntradayConfig) -> pd.DataFrame:
    bpy = bars_per_year(cfg.bar)
    return size_weights(signal.reindex(d.index), d.panel(), intraday_portfolio_config(cfg), bpy, "perp",
                        min_history_bars=288 * 7)


def backtest_weights(w: pd.DataFrame, d: IntradayData, cfg: IntradayConfig, start=None, end=None,
                     extra_bps: float = 0.0) -> BacktestResult:
    bps = np.array(cfg.cost_bps(list(w.columns))) + extra_bps
    return run_backtest(w, d.panel(), CostConfig(0, 0), band=cfg.rebalance_band, start=start, end=end,
                        cost_bps=bps)


def daily_metrics(res: BacktestResult) -> dict:
    """Metrics on daily-compounded returns (avoids intraday autocorrelation artefacts)."""
    r = res.returns
    if r.empty:
        return {}
    daily = (1.0 + r).groupby(r.index.floor("D")).prod() - 1.0
    m = summarize(daily, 365.0)
    bars = len(r)
    per_year = bars_per_year("5m")
    m["turnover_ann"] = float(res.turnover.mean() * per_year)
    m["cost_drag_ann"] = float(res.costs.mean() * per_year)
    m["funding_drag_ann"] = float(res.funding.mean() * per_year)
    m["gross_ret_ann"] = float(res.gross.mean() * per_year)
    m["avg_gross"] = float(res.weights.abs().sum(axis=1).mean())
    m["avg_net"] = float(res.weights.sum(axis=1).mean())
    m["profitable_days"] = float((daily > 0).mean())
    m["bars"] = bars
    return m


def information_coefficient(pred: pd.DataFrame, fs: FeatureSet) -> float:
    """Average daily cross-pooled rank correlation between forecasts and realised targets."""
    y = pd.DataFrame(fs.y, index=fs.index, columns=fs.symbols).reindex(pred.index)
    p = pred.stack()
    t = y.stack().reindex(p.index)
    ok = p.notna() & t.notna()
    if ok.sum() < 100:
        return float("nan")
    df = pd.DataFrame({"p": p[ok], "t": t[ok]})
    day = df.index.get_level_values(0).floor("D")
    ics = df.groupby(day).apply(lambda g: g["p"].rank().corr(g["t"].rank()))
    return float(ics.mean())


def annualised(x: float) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{100 * x:.1f}%"


# ------------------------------------------------- selective (hysteresis) rule
def _hysteresis_py(score: np.ndarray, enter: float, exit_: float, min_hold: int) -> np.ndarray:
    T, N = score.shape
    pos = np.zeros((T, N))
    for i in range(N):
        p = 0.0
        held = 0
        for t in range(T):
            z = score[t, i]
            if np.isnan(z):
                p = 0.0
                held = 0
            elif p == 0.0:
                if z > enter:
                    p, held = 1.0, 0
                elif z < -enter:
                    p, held = -1.0, 0
            else:
                held += 1
                if held >= min_hold:
                    if p > 0 and z < exit_:
                        p = -1.0 if z < -enter else 0.0
                        held = 0
                    elif p < 0 and z > -exit_:
                        p = 1.0 if z > enter else 0.0
                        held = 0
            pos[t, i] = p
    return pos


_HYST = None


def hysteresis_positions(score: pd.DataFrame, enter: float, exit_: float, min_hold: int = 1) -> pd.DataFrame:
    """Discrete {-1, 0, +1} positions: enter beyond ±enter, leave when the score falls back
    through ±exit (after at least ``min_hold`` bars), flip directly on an opposite entry signal."""
    global _HYST
    if _HYST is None:
        try:
            import numba

            _HYST = numba.njit(cache=True)(_hysteresis_py)
        except Exception:  # noqa: BLE001
            _HYST = _hysteresis_py
    out = _HYST(score.to_numpy(dtype="float64"), float(enter), float(exit_), int(min_hold))
    return pd.DataFrame(out, index=score.index, columns=score.columns)


def forecast_score(pred: pd.DataFrame, horizon: int, norm_window: int = 288 * 30) -> pd.DataFrame:
    """Forecast averaged over its horizon, in units of its trailing mean absolute value (causal)."""
    s = pred.rolling(horizon, min_periods=1).mean()
    return s / s.abs().rolling(norm_window, min_periods=288).mean()


def freeze_between_changes(pos: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """Keep each position's weight fixed from entry until the position changes (no re-sizing churn)."""
    p = pos.to_numpy()
    W = w.reindex(index=pos.index, columns=pos.columns).fillna(0.0).to_numpy()
    out = np.zeros_like(W)
    for i in range(p.shape[1]):
        cur = 0.0
        for t in range(p.shape[0]):
            if p[t, i] == 0.0:
                cur = 0.0
            elif t == 0 or p[t, i] != p[t - 1, i] or cur == 0.0:
                cur = W[t, i]
            out[t, i] = cur
    return pd.DataFrame(out, index=pos.index, columns=pos.columns)


def selective_weights(pred: pd.DataFrame, d: IntradayData, cfg: IntradayConfig, horizon: int,
                      enter: float, exit_: float, min_hold: int, norm_window: int = 288 * 30) -> pd.DataFrame:
    score = forecast_score(pred.reindex(d.index), horizon, norm_window)
    pos = hysteresis_positions(score, enter, exit_, min_hold)
    w = signal_to_weights(pos.where(score.notna()), d, cfg)
    return freeze_between_changes(pos, w)
