"""End-to-end deep intraday traders (Deep-Momentum-Network style).

Instead of forecasting returns and then hand-tuning a trading rule, the network
outputs a position in [-1, 1] for every 5-minute bar and is trained to maximise
the Sharpe ratio of *net* returns: position × next-bar return minus the real
cost of every change in position. Costs therefore shape the policy directly;
the model learns to trade only when the expected gain exceeds the fee.

Models (shared across assets, asset embedding as static context):
* ``lstm``      – LSTM deep momentum network (Lim et al. 2019);
* ``lstm_attn`` – Momentum-Transformer-lite: LSTM encoder + causal multi-head
                  self-attention with gated residuals (Wood et al. 2021), without
                  the per-variable selection network to stay CPU-friendly at 5m.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.config import bars_per_year, utc
from tradinglab.intraday.data import IntradayData
from tradinglab.intraday.features import FeatureSet

log = logging.getLogger(__name__)


@dataclass
class DeepIntradayConfig:
    kind: str = "lstm_attn"
    seq_len: int = 96          # 8 hours of 5m bars
    burn_in: int = 24
    hidden: int = 32
    heads: int = 4
    dropout: float = 0.2
    lr: float = 1e-3
    batch_windows: int = 16    # time windows per batch (× assets sequences)
    train_stride: int = 24
    max_epochs: int = 12
    patience: int = 3
    cost_bps: float = 7.0      # average taker fee + slippage per side
    turnover_penalty: float = 1.0
    seeds: int = 2
    threads: int = 4
    asset_vol_target: float = 0.6


def _torch():
    import torch

    return torch


def build_net(kind: str, n_feat: int, n_assets: int, cfg: DeepIntradayConfig):
    torch = _torch()
    from torch import nn

    from tradinglab.strategies.deep.layers import AttentionBlock, GateAddNorm, causal_mask

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            h = cfg.hidden
            self.inp = nn.Sequential(nn.Linear(n_feat, h), nn.ELU(), nn.Dropout(cfg.dropout))
            self.asset = nn.Embedding(n_assets, h)
            self.init_h = nn.Linear(h, h)
            self.init_c = nn.Linear(h, h)
            self.lstm = nn.LSTM(h, h, batch_first=True)
            self.gate = GateAddNorm(h, dropout=cfg.dropout)
            self.attn = AttentionBlock(h, cfg.heads, cfg.dropout) if kind == "lstm_attn" else None
            self.head = nn.Sequential(nn.Linear(h, 1), nn.Tanh())

        def forward(self, x, mask):
            B, N, L, F = x.shape
            ids = torch.arange(N, device=x.device).repeat(B)
            flat = x.reshape(B * N, L, F)
            s = self.asset(ids)
            z = self.inp(flat)
            out, _ = self.lstm(z, (torch.tanh(self.init_h(s)).unsqueeze(0), torch.tanh(self.init_c(s)).unsqueeze(0)))
            out = self.gate(out, z)
            if self.attn is not None:
                out = self.attn(out, attn_mask=causal_mask(L, x.device))
            pos = self.head(out).squeeze(-1).reshape(B, N, L)
            return pos * mask.unsqueeze(-1)

    return Net()


@dataclass
class DeepData:
    X: np.ndarray       # [T, N, F] with NaN
    Y: np.ndarray       # [T, N] vol-scaled next-bar return
    S: np.ndarray       # [T, N] position → notional scale
    ok: np.ndarray      # [T, N] row usable


def prepare(fs: FeatureSet, d: IntradayData, cfg: DeepIntradayConfig) -> DeepData:
    bpy = bars_per_year(d.bar)
    ex = d.exec_price()
    r1 = (ex.shift(-1) / ex - 1.0).to_numpy(dtype="float64")
    sigma_ann = fs.sigma * math.sqrt(bpy)
    S = np.clip(cfg.asset_vol_target / sigma_ann, 0, 20).astype("float32")
    Y = np.clip(r1 * S, -0.5, 0.5).astype("float32")
    ok = np.isfinite(fs.X).all(axis=-1) & np.isfinite(S)
    return DeepData(fs.X, Y, S, ok)


def _window_ok(dd: DeepData, ends: np.ndarray, L: int, need_y: bool) -> np.ndarray:
    ok = dd.ok & np.isfinite(dd.Y) if need_y else dd.ok
    c = np.concatenate([np.zeros((1, ok.shape[1]), dtype=np.int64), np.cumsum(ok, axis=0)])
    starts = ends - L + 1
    v = (c[ends + 1] - c[np.maximum(starts, 0)]) == L
    v[starts < 0] = False
    return v


def _batch(dd: DeepData, ends: np.ndarray, L: int, need_y: bool):
    torch = _torch()
    rows = ends[:, None] + np.arange(-L + 1, 1)[None, :]
    valid = _window_ok(dd, ends, L, need_y)
    x = np.transpose(dd.X[rows], (0, 2, 1, 3))
    x = np.where(valid[:, :, None, None], np.nan_to_num(x), 0.0).astype("float32")
    y = np.transpose(np.nan_to_num(dd.Y[rows]), (0, 2, 1)).astype("float32")
    s = np.transpose(np.nan_to_num(dd.S[rows]), (0, 2, 1)).astype("float32")
    return (torch.from_numpy(x), torch.from_numpy(valid.astype("float32")), torch.from_numpy(y),
            torch.from_numpy(s))


class DeepIntradayTrader:
    def __init__(self, cfg: DeepIntradayConfig, bar: str = "5m"):
        self.cfg = cfg
        self.bpy = bars_per_year(bar)
        self.states: list[dict] = []
        self.n_feat = self.n_assets = None

    def _loss(self, net, batch, train: bool):
        from tradinglab.strategies.deep.trainer import sharpe_objective, strategy_returns

        x, mask, y, s = batch
        pos = net(x, mask)
        ret = strategy_returns(pos, y, s, mask, self.cfg.cost_bps * 1e-4 * self.cfg.turnover_penalty)
        return sharpe_objective(ret, mask, self.cfg.burn_in, "asset", self.bpy)

    def fit(self, dd: DeepData, index: pd.DatetimeIndex, train_start: pd.Timestamp, train_end: pd.Timestamp,
            valid_days: int = 30) -> list[dict]:
        torch = _torch()
        torch.set_num_threads(self.cfg.threads)
        L = self.cfg.seq_len
        lo = int(np.searchsorted(index, utc(train_start)))
        hi = int(np.searchsorted(index, utc(train_end))) - 2
        vs = int(np.searchsorted(index, utc(train_end) - pd.Timedelta(days=valid_days)))
        train = np.arange(max(lo + L, L - 1), vs - L, self.cfg.train_stride)
        val = np.arange(vs, hi, L - self.cfg.burn_in)
        train = train[_window_ok(dd, train, L, True).sum(axis=1) > 0]
        val = val[_window_ok(dd, val, L, True).sum(axis=1) > 0]
        self.n_feat, self.n_assets = dd.X.shape[-1], dd.X.shape[1]
        val_batch = _batch(dd, val, L, True)
        self.states = []
        for seed in range(self.cfg.seeds):
            torch.manual_seed(seed)
            net = build_net(self.cfg.kind, self.n_feat, self.n_assets, self.cfg)
            opt = torch.optim.Adam(net.parameters(), lr=self.cfg.lr)
            rng = np.random.default_rng(seed)
            best, best_state, bad = -np.inf, None, 0
            for epoch in range(self.cfg.max_epochs):
                net.train()
                order = rng.permutation(train)
                for i in range(0, len(order), self.cfg.batch_windows):
                    loss = self._loss(net, _batch(dd, np.sort(order[i:i + self.cfg.batch_windows]), L, True), True)
                    if not torch.isfinite(loss):
                        continue
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = -float(self._loss(net, val_batch, False))
                if v > best + 1e-3:
                    best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= self.cfg.patience:
                        break
            log.info("%s seed %d: val net Sharpe %.2f after %d epochs", self.cfg.kind, seed, best, epoch + 1)
            self.states.append({"seed": seed, "val_sharpe": best, "state": best_state})
        return self.states

    def predict(self, dd: DeepData, ends: np.ndarray, chunk: int = 256) -> np.ndarray:
        torch = _torch()
        torch.set_num_threads(self.cfg.threads)
        L = self.cfg.seq_len
        out = np.full((len(ends), dd.X.shape[1]), np.nan)
        nets = []
        for st in self.states:
            net = build_net(self.cfg.kind, self.n_feat, self.n_assets, self.cfg)
            net.load_state_dict(st["state"])
            net.eval()
            nets.append(net)
        with torch.no_grad():
            for i in range(0, len(ends), chunk):
                e = ends[i:i + chunk]
                x, mask, _, _ = _batch(dd, e, L, need_y=False)
                pos = torch.stack([n(x, mask)[..., -1] for n in nets]).mean(0).numpy()
                out[i:i + chunk] = np.where(mask.numpy() > 0, pos, np.nan)
        return out

    def save(self, path: Path) -> None:
        torch = _torch()
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.states, path / "weights.pt")
        (path / "meta.json").write_text(json.dumps({"cfg": self.cfg.__dict__, "n_feat": self.n_feat,
                                                     "n_assets": self.n_assets}, indent=1))

    def load(self, path: Path) -> "DeepIntradayTrader":
        torch = _torch()
        meta = json.loads((path / "meta.json").read_text())
        self.cfg = DeepIntradayConfig(**meta["cfg"])
        self.n_feat, self.n_assets = meta["n_feat"], meta["n_assets"]
        self.states = torch.load(path / "weights.pt", weights_only=False)
        return self


def walk_forward_deep(fs: FeatureSet, d: IntradayData, cfg: DeepIntradayConfig, first_test: str,
                      train_days: int = 365, step_months: int = 3) -> pd.DataFrame:
    """OOS positions in [-1, 1]; the model is retrained every ``step_months``."""
    from tradinglab.intraday.research import month_starts

    dd = prepare(fs, d, cfg)
    idx = fs.index
    starts = month_starts(idx, first_test, step_months)
    parts = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else idx[-1] + pd.Timedelta(minutes=5)
        trader = DeepIntradayTrader(cfg, d.bar)
        trader.fit(dd, idx, s - pd.Timedelta(days=train_days), s)
        lo, hi = int(np.searchsorted(idx, s)), int(np.searchsorted(idx, e))
        ends = np.arange(lo, hi)
        parts.append(pd.DataFrame(trader.predict(dd, ends), index=idx[lo:hi], columns=fs.symbols))
        log.info("deep fold %s done", s.date())
    return pd.concat(parts).sort_index()
