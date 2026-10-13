"""Combine several strategies' weights into one portfolio (used by research and live)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.config import LabConfig, bars_per_year
from tradinglab.data.store import Panel
from tradinglab.portfolio.sizing import vol_target


def strategy_alphas(oos_returns: dict[str, pd.Series], method: str = "inverse_vol") -> dict[str, float]:
    """Capital split between strategies from their out-of-sample return streams."""
    names = list(oos_returns)
    if not names:
        return {}
    if method == "equal":
        raw = {n: 1.0 for n in names}
    elif method == "inverse_vol":
        raw = {n: 1.0 / max(float(oos_returns[n].std()), 1e-6) for n in names}
    elif method == "sharpe":
        raw = {n: max(float(oos_returns[n].mean() / max(oos_returns[n].std(), 1e-9)), 0.0) for n in names}
        if sum(raw.values()) <= 0:
            raw = {n: 1.0 for n in names}
    else:
        raise ValueError(f"unknown weighting {method!r}")
    total = sum(raw.values())
    return {n: v / total for n, v in raw.items()}


def combine_weights(weights: dict[str, pd.DataFrame], alphas: dict[str, float], panel: Panel,
                    cfg: LabConfig) -> pd.DataFrame:
    """α-weighted sum of strategy weights, re-targeted to the portfolio vol target."""
    names = [n for n in alphas if n in weights]
    if not names:
        return pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
    cols = panel.close.columns
    total = sum(alphas[n] * weights[n].reindex(index=panel.close.index, columns=cols).fillna(0.0)
                for n in names)
    if len(names) == 1:
        return total
    return vol_target(total, panel, cfg.portfolio, bars_per_year(cfg.market.bar), cfg.market.instrument)


def correlation_filter(returns: dict[str, pd.Series], ranked: list[str], max_corr: float,
                       limit: int) -> list[str]:
    """Greedy pick in rank order, skipping strategies too correlated with ones already chosen."""
    chosen: list[str] = []
    for name in ranked:
        if len(chosen) >= limit:
            break
        ok = True
        for c in chosen:
            a, b = returns[name].align(returns[c], join="inner")
            if len(a) > 30 and np.corrcoef(a, b)[0, 1] > max_corr:
                ok = False
                break
        if ok:
            chosen.append(name)
    return chosen
