"""Performance and statistical-significance metrics.

Includes the Probabilistic Sharpe Ratio and Deflated Sharpe Ratio of Bailey &
López de Prado (2012, 2014) so strategy selection accounts for estimation error,
non-normal returns and the number of strategies tried.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe(r: pd.Series | np.ndarray, periods: float) -> float:
    r = np.asarray(pd.Series(r).dropna(), dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period SR > sr_benchmark) given sample moments (Bailey & LdP 2012)."""
    x = np.asarray(pd.Series(r).dropna(), dtype=float)
    n = len(x)
    if n < 10 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe estimates under the null."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(max(sr_variance, 0.0)) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def effective_trials(returns: pd.DataFrame) -> float:
    """Effective number of independent trials, M_eff = ρ̄ + (1 - ρ̄)·M.

    ρ̄ is the average pairwise correlation of the trials' return streams. Highly
    correlated variants (e.g. several trend filters) count as fewer independent
    attempts than structurally different strategies.
    """
    m = returns.shape[1]
    if m <= 1:
        return float(m)
    c = returns.corr().to_numpy()
    rho = float(np.nanmean(c[~np.eye(m, dtype=bool)]))
    rho = min(max(rho, 0.0), 1.0)
    return rho + (1.0 - rho) * m


def deflated_sharpe(r: pd.Series, trial_sharpes_per_period: list[float], n_trials: float | None = None) -> float:
    """DSR: PSR against the expected max Sharpe of ``n_trials`` trials (per-period units).

    The dispersion of the trials' Sharpe ratios estimates the null variance; ``n_trials``
    defaults to the number of trials but should be the *effective* number when the
    trials are correlated (see :func:`effective_trials`).
    """
    trials = [s for s in trial_sharpes_per_period if np.isfinite(s)]
    n = n_trials if n_trials is not None else max(len(trials), 1)
    var = float(np.var(trials, ddof=1)) if len(trials) > 1 else 0.0
    return probabilistic_sharpe(r, expected_max_sharpe(var, max(int(round(n)), 1)))


def newey_west_tstat(r: pd.Series, lags: int | None = None) -> float:
    """HAC (Newey-West) t-statistic of the mean return."""
    x = np.asarray(pd.Series(r).dropna(), dtype=float)
    n = len(x)
    if n < 10:
        return float("nan")
    lags = lags if lags is not None else int(4 * (n / 100) ** (2 / 9))
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    se = math.sqrt(max(s, 1e-18) / n)
    return float(x.mean() / se)


def summarize(r: pd.Series, periods: float, turnover: pd.Series | None = None,
              costs: pd.Series | None = None) -> dict[str, float]:
    r = pd.Series(r).dropna()
    if r.empty:
        return {}
    eq = (1.0 + r).cumprod()
    years = len(r) / periods
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and eq.iloc[-1] > 0 else float("nan")
    vol = float(r.std(ddof=1) * math.sqrt(periods))
    downside = r[r < 0]
    sortino = (float(r.mean() / downside.std(ddof=1) * math.sqrt(periods))
               if len(downside) > 2 and downside.std(ddof=1) > 0 else float("nan"))
    mdd = max_drawdown(r)
    out = {
        "start": str(r.index[0].date()) if hasattr(r.index[0], "date") else str(r.index[0]),
        "end": str(r.index[-1].date()) if hasattr(r.index[-1], "date") else str(r.index[-1]),
        "bars": int(len(r)),
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe(r, periods),
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else float("nan"),
        "hit_rate": float((r > 0).mean()),
        "skew": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r)),
        "psr": probabilistic_sharpe(r),
        "nw_tstat": newey_west_tstat(r),
    }
    if turnover is not None:
        out["turnover_ann"] = float(turnover.reindex(r.index).fillna(0).mean() * periods)
    if costs is not None:
        out["cost_drag_ann"] = float(costs.reindex(r.index).fillna(0).mean() * periods)
    return out
