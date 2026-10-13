"""Causal signal → weight conversion shared by research and live trading.

1. Per-asset volatility normalisation: ``w_i = s_i · σ_target / σ_i / N_active``
   (the time-series-momentum convention of Moskowitz et al. and the deep
   momentum papers).
2. Portfolio volatility targeting with an EWMA covariance estimate.
3. Hard caps on per-asset weight and gross leverage; long-only for spot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.config import PortfolioConfig
from tradinglab.data.store import Panel


def history_mask(close: pd.DataFrame, min_bars: int) -> pd.DataFrame:
    """True where an asset has at least ``min_bars`` bars of history (listing seasoning)."""
    return close.notna().cumsum() >= min_bars


def ewma_cov(returns: np.ndarray, span: int) -> np.ndarray:
    """Recursive EWMA covariance, returns array [T, N, N] (causal: uses rows <= t)."""
    lam = 1.0 - 2.0 / (span + 1.0)
    T, N = returns.shape
    r = np.nan_to_num(returns, nan=0.0)
    out = np.empty((T, N, N))
    S = np.zeros((N, N))
    for t in range(T):
        S = lam * S + (1.0 - lam) * np.outer(r[t], r[t])
        out[t] = S
    return out


def size_weights(signals: pd.DataFrame, panel: Panel, cfg: PortfolioConfig, bars_per_year: float,
                 instrument: str = "perp", min_history_bars: int = 0,
                 outputs_weights: bool = False, smoothing_span: int = 1) -> pd.DataFrame:
    close = panel.close.reindex(columns=signals.columns)
    sig = signals.reindex(close.index)
    if smoothing_span > 1:
        sig = sig.ewm(span=smoothing_span, min_periods=1).mean().where(sig.notna())
    rets = close.pct_change(fill_method=None)
    ann = np.sqrt(bars_per_year)
    vol = rets.ewm(span=cfg.vol_lookback, min_periods=cfg.vol_lookback // 2).std() * ann

    tradable = close.notna() & vol.notna() & (vol > 0)
    if min_history_bars:
        tradable &= history_mask(close, min_history_bars)
    sig = sig.where(tradable)
    if instrument == "spot":
        sig = sig.clip(lower=0.0)

    if outputs_weights:
        raw = sig.fillna(0.0)
    else:
        n_active = tradable.sum(axis=1).clip(lower=1)
        raw = (sig * (cfg.asset_vol_target / vol)).div(n_active, axis=0).fillna(0.0)

    return vol_target(raw, panel, cfg, bars_per_year, instrument)


def strategy_weights(strategy, signals: pd.DataFrame, panel: Panel, lab_cfg) -> pd.DataFrame:
    """Signals → weights with the lab's settings (the single path used by research *and* live)."""
    from tradinglab.config import bars_per_year

    span = lab_cfg.portfolio.signal_span if getattr(strategy, "smooth_signals", False) else 1
    return size_weights(signals.reindex(panel.close.index), panel, lab_cfg.portfolio,
                        bars_per_year(lab_cfg.market.bar), lab_cfg.market.instrument,
                        lab_cfg.universe.min_history_bars, strategy.outputs_weights, smoothing_span=span)


def vol_target(raw: pd.DataFrame, panel: Panel, cfg: PortfolioConfig, bars_per_year: float,
               instrument: str = "perp") -> pd.DataFrame:
    """Scale raw weights to the portfolio vol target (EWMA covariance), then apply caps."""
    close = panel.close.reindex(index=raw.index, columns=raw.columns)
    rets = close.pct_change(fill_method=None)
    ann = np.sqrt(bars_per_year)
    cov = ewma_cov(rets.to_numpy(), cfg.vol_lookback)
    W = raw.fillna(0.0).to_numpy()
    port_var = np.einsum("ti,tij,tj->t", W, cov, W)
    port_vol = np.sqrt(np.maximum(port_var, 0.0)) * ann
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(port_vol > 1e-8, cfg.target_vol / port_vol, 0.0)
    # Warm-up: covariance needs history before it is trusted.
    scale[: cfg.vol_lookback] = 0.0
    W = W * scale[:, None]
    W = np.clip(W, -cfg.max_weight, cfg.max_weight)
    gross = np.abs(W).sum(axis=1)
    shrink = np.where(gross > cfg.max_gross, cfg.max_gross / np.maximum(gross, 1e-12), 1.0)
    W = W * shrink[:, None]
    if instrument == "spot":
        W = np.clip(W, 0.0, None)
    return pd.DataFrame(W, index=raw.index, columns=raw.columns)
