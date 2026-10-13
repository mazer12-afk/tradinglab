"""Data preparation, Sharpe-ratio training and inference for deep momentum models.

Training directly maximises the (annualised) Sharpe ratio of strategy returns
net of an estimated transaction cost, as in the Deep Momentum Network family.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradinglab.config import DeepConfig, utc
from tradinglab.data.store import Panel
from tradinglab.features import DEEP_FEATURES, deep_features, stack_features
from tradinglab.portfolio.sizing import history_mask

log = logging.getLogger(__name__)

TARGET_CLIP = 5.0


@dataclass
class DeepData:
    index: pd.DatetimeIndex
    symbols: list[str]
    X: np.ndarray        # [T, N, F] features (NaN where unavailable)
    Y: np.ndarray        # [T, N]   vol-scaled next-period return
    S: np.ndarray        # [T, N]   vol scaling σ_target/σ (position → notional)
    feat_ok: np.ndarray  # [T, N]   usable inputs at t
    tgt_ok: np.ndarray   # [T, N]   target known

    def window_ok(self, ends: np.ndarray, L: int, need_target: bool) -> np.ndarray:
        """[len(ends), N] – asset fully valid over each window."""
        ok = self.feat_ok & self.tgt_ok if need_target else self.feat_ok
        c = np.concatenate([np.zeros((1, ok.shape[1]), dtype=np.int64), np.cumsum(ok, axis=0)])
        starts = ends - L + 1
        valid = (c[ends + 1] - c[np.maximum(starts, 0)]) == L
        valid[starts < 0] = False
        return valid


def prepare_data(panel: Panel, asset_vol_target: float, vol_lookback: int, bars_per_year: float,
                 min_history_bars: int, features: list[str] = DEEP_FEATURES) -> DeepData:
    close = panel.close
    feats = deep_features(close)
    X = stack_features(feats, features)
    rets = close.pct_change(fill_method=None)
    vol = rets.ewm(span=vol_lookback, min_periods=vol_lookback // 2).std() * math.sqrt(bars_per_year)
    scale = (asset_vol_target / vol).clip(upper=10.0)
    Y = (panel.returns() * scale).clip(-TARGET_CLIP, TARGET_CLIP)
    hist = history_mask(close, min_history_bars).to_numpy()
    S = scale.to_numpy(dtype="float32")
    feat_ok = np.isfinite(X).all(axis=-1) & np.isfinite(S) & hist
    Yn = Y.to_numpy(dtype="float32")
    return DeepData(close.index, list(close.columns), X, Yn, S, feat_ok, np.isfinite(Yn))


def _torch():
    import torch

    return torch


def _batch(data: DeepData, ends: np.ndarray, L: int, need_target: bool):
    torch = _torch()
    offs = np.arange(-L + 1, 1)
    rows = ends[:, None] + offs[None, :]                     # [B, L]
    valid = data.window_ok(ends, L, need_target)              # [B, N]
    x = data.X[rows]                                          # [B, L, N, F]
    x = np.transpose(x, (0, 2, 1, 3))                         # [B, N, L, F]
    x = np.where(valid[:, :, None, None], np.nan_to_num(x), 0.0).astype("float32")
    y = np.transpose(np.nan_to_num(data.Y[rows]), (0, 2, 1)).astype("float32")
    s = np.transpose(np.nan_to_num(data.S[rows]), (0, 2, 1)).astype("float32")
    return (torch.from_numpy(x), torch.from_numpy(valid.astype("float32")),
            torch.from_numpy(y), torch.from_numpy(s))


def strategy_returns(pos, y, s, mask, cost_rate: float):
    """Per-asset net returns [B, N, L] of positions ``pos`` (vol-target units)."""
    torch = _torch()
    w = pos * s
    turnover = torch.zeros_like(w)
    turnover[..., 1:] = (w[..., 1:] - w[..., :-1]).abs()
    return (pos * y - cost_rate * turnover) * mask.unsqueeze(-1)


def sharpe_objective(ret, mask, burn_in: int, level: str, bpy: float,
                     softmin_lambda: float = 0.0, softmin_beta: float = 5.0):
    """Negative annualised Sharpe (+ optional soft-min worst-window term)."""
    torch = _torch()
    ret = ret[..., burn_in:]
    ann = math.sqrt(bpy)
    if level == "asset":
        m = mask.unsqueeze(-1).expand_as(ret)
        vals = ret[m > 0]
        loss = -vals.mean() / (vals.std() + 1e-8) * ann
        return loss
    n = mask.sum(dim=1).clamp(min=1.0)                      # [B]
    port = ret.sum(dim=1) / n.unsqueeze(-1)                 # [B, L']
    keep = mask.sum(dim=1) > 0
    port = port[keep]
    pooled = port.reshape(-1)
    loss = -pooled.mean() / (pooled.std() + 1e-8) * ann
    if softmin_lambda > 0 and port.shape[0] > 1:
        sr_b = port.mean(dim=1) / (port.std(dim=1) + 1e-8) * ann
        softmin = -torch.logsumexp(-softmin_beta * sr_b, dim=0) / softmin_beta
        loss = loss - softmin_lambda * softmin
    return loss


class DeepTrainer:
    def __init__(self, kind: str, cfg: DeepConfig, cost_bps: float, bars_per_year: float):
        self.kind = kind
        self.cfg = cfg
        self.cost_rate = cost_bps * 1e-4
        self.bpy = bars_per_year
        self.level = "portfolio" if kind == "deepm" else "asset"
        self.train_stride = 10
        self.softmin_lambda = cfg.softmin_lambda if kind == "deepm" else 0.0

    def _new_model(self, n_features: int, n_assets: int, seed: int):
        torch = _torch()
        from tradinglab.strategies.deep.models import build_model

        torch.manual_seed(seed)
        np.random.seed(seed)
        return build_model(self.kind, num_features=n_features, num_assets=n_assets,
                           hidden=self.cfg.hidden, dropout=self.cfg.dropout, heads=self.cfg.heads)

    def _loss(self, model, batch, train: bool):
        x, mask, y, s = batch
        pos = model(x, mask)
        ret = strategy_returns(pos, y, s, mask, self.cost_rate * self.cfg.turnover_penalty)
        return sharpe_objective(ret, mask, self.cfg.burn_in, self.level, self.bpy,
                                self.softmin_lambda if train else 0.0, self.cfg.softmin_beta)

    def split(self, data: DeepData, train_end: pd.Timestamp,
              valid_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        L = self.cfg.seq_len
        t_end = int(np.searchsorted(data.index, utc(train_end)))  # first index >= train_end
        last = t_end - 2                                      # target at e needs bar e+1 < train_end
        ends = np.arange(L - 1, last + 1)
        ok = data.window_ok(ends, L, need_target=True).sum(axis=1) >= 2
        ends = ends[ok]
        if len(ends) < 20:
            raise ValueError(f"not enough training windows before {train_end}: {len(ends)}")
        n_val = max(int(len(ends) * valid_fraction), 5)
        val_start = ends[-n_val]
        train = ends[ends <= val_start - L][:: self.train_stride]
        val = ends[ends >= val_start][:: max(1, (L - self.cfg.burn_in) // 2)]
        return train, val

    def fit(self, data: DeepData, train_end: pd.Timestamp, seeds: int | None = None,
            valid_fraction: float = 0.15) -> list[dict]:
        torch = _torch()
        torch.set_num_threads(self.cfg.threads)
        train, val = self.split(data, train_end, valid_fraction)
        L = self.cfg.seq_len
        val_batch = _batch(data, val, L, True)
        states = []
        for seed in range(seeds or self.cfg.seeds):
            model = self._new_model(data.X.shape[-1], data.X.shape[1], seed)
            opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
            rng = np.random.default_rng(seed)
            best, best_state, bad = -np.inf, None, 0
            bs = max(4, self.cfg.batch_size // max(1, data.X.shape[1] // 4))
            for epoch in range(self.cfg.max_epochs):
                model.train()
                order = rng.permutation(train)
                for i in range(0, len(order), bs):
                    batch = _batch(data, np.sort(order[i:i + bs]), L, True)
                    loss = self._loss(model, batch, train=True)
                    if not torch.isfinite(loss):
                        continue
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vs = -float(self._loss(model, val_batch, train=False))
                if vs > best + 1e-4:
                    best, best_state, bad = vs, copy.deepcopy(model.state_dict()), 0
                else:
                    bad += 1
                    if bad >= self.cfg.patience:
                        break
            log.info("%s seed %d: best val Sharpe %.2f after %d epochs", self.kind, seed, best, epoch + 1)
            states.append({"seed": seed, "val_sharpe": best, "state": best_state})
        return states

    def predict(self, data: DeepData, states: list[dict], ends: np.ndarray, chunk: int = 64) -> np.ndarray:
        """Ensemble-average positions [len(ends), N] (NaN where an asset has no valid window)."""
        torch = _torch()
        torch.set_num_threads(self.cfg.threads)
        L = self.cfg.seq_len
        out = np.full((len(ends), data.X.shape[1]), np.nan, dtype="float64")
        if len(ends) == 0 or not states:
            return out
        models = []
        for st in states:
            m = self._new_model(data.X.shape[-1], data.X.shape[1], st["seed"])
            m.load_state_dict(st["state"])
            m.eval()
            models.append(m)
        with torch.no_grad():
            for i in range(0, len(ends), chunk):
                e = ends[i:i + chunk]
                ok_e = e >= L - 1
                if not ok_e.any():
                    continue
                x, mask, _, _ = _batch(data, e[ok_e], L, need_target=False)
                pos = torch.stack([m(x, mask)[..., -1] for m in models]).mean(0).numpy()
                pos = np.where(mask.numpy() > 0, pos, np.nan)
                out[i:i + chunk][ok_e] = pos
        return out
