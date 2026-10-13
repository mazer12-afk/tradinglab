"""Deep momentum network architectures.

All models consume a cross-section of sequences ``x`` of shape [B, N, L, F]
(batch of time windows × assets × time steps × features) plus an asset mask
[B, N] and return positions in [-1, 1] of shape [B, N, L]. Per-asset models
(LSTM, Momentum Transformer) process each asset independently; DeePM-lite adds
attention across assets.

* ``lstm``    – Deep Momentum Network (Lim, Zohren & Roberts, 2019).
* ``mom_trans`` – Momentum Transformer (Wood, Giegerich, Roberts & Zohren, 2021):
                VSN → LSTM → causal self-attention with TFT gating.
* ``deepm``   – DeePM-lite (after Wood, Roberts & Zohren, 2026): temporal
                backbone + cross-sectional attention with ReZero gating, trained
                on portfolio-level Sharpe with a soft-min worst-window penalty.
                The macro graph prior is omitted (no economic graph for crypto);
                bars close simultaneously for crypto so same-time cross-attention
                is causal (the paper's lag exists for asynchronous futures closes).
"""

from __future__ import annotations

import torch
from torch import nn

from tradinglab.strategies.deep.layers import (
    AttentionBlock,
    GateAddNorm,
    GatedResidualNetwork,
    ResSwiGLU,
    VariableSelectionNetwork,
    causal_mask,
)


class DeepMomentumModel(nn.Module):
    cross_sectional = False

    def __init__(self, num_features: int, num_assets: int, hidden: int, dropout: float, **_):
        super().__init__()
        self.num_features = num_features
        self.num_assets = num_assets
        self.hidden = hidden
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())

    def encode(self, x: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:  # [M, L, F] → [M, L, H]
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, L, F = x.shape
        ids = torch.arange(N, device=x.device).repeat(B)
        flat_mask = mask.reshape(-1) > 0
        # Encode only assets present in each window (not-yet-listed assets are padding).
        enc = self.encode(x.reshape(B * N, L, F)[flat_mask], ids[flat_mask])
        h = enc.new_zeros(B * N, L, enc.shape[-1])
        h[flat_mask] = enc
        h = self.cross_section(h.reshape(B, N, L, -1), mask)
        pos = self.head(h).squeeze(-1)  # [B, N, L]
        return pos * mask.unsqueeze(-1)

    def cross_section(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return h


class LSTMDMN(DeepMomentumModel):
    """LSTM deep momentum network (Lim et al., 2019)."""

    def __init__(self, num_features, num_assets, hidden, dropout, **kw):
        super().__init__(num_features, num_assets, hidden, dropout)
        self.lstm = nn.LSTM(num_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def encode(self, x, asset_ids):
        out, _ = self.lstm(x)
        return self.drop(out)


class MomentumTransformer(DeepMomentumModel):
    """Momentum Transformer: VSN + LSTM encoder + causal interpretable attention."""

    def __init__(self, num_features, num_assets, hidden, dropout, heads=4, **kw):
        super().__init__(num_features, num_assets, hidden, dropout)
        self.asset_emb = nn.Embedding(num_assets, hidden)
        self.ctx_vsn = GatedResidualNetwork(hidden, hidden, hidden, dropout)
        self.ctx_h = GatedResidualNetwork(hidden, hidden, hidden, dropout)
        self.ctx_c = GatedResidualNetwork(hidden, hidden, hidden, dropout)
        self.ctx_enrich = GatedResidualNetwork(hidden, hidden, hidden, dropout)
        self.vsn = VariableSelectionNetwork(num_features, hidden, dropout, context_size=hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm_gate = GateAddNorm(hidden, dropout=dropout)
        self.enrich = GatedResidualNetwork(hidden, hidden, hidden, dropout, context_size=hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout)
        self.attn_gate = GateAddNorm(hidden, dropout=dropout)
        self.ffn = GatedResidualNetwork(hidden, hidden, hidden, dropout)
        self.out_gate = GateAddNorm(hidden, dropout=dropout)

    def encode(self, x, asset_ids):
        s = self.asset_emb(asset_ids)
        v, _ = self.vsn(x, self.ctx_vsn(s))
        h0 = self.ctx_h(s).unsqueeze(0)
        c0 = self.ctx_c(s).unsqueeze(0)
        lstm_out, _ = self.lstm(v, (h0, c0))
        temporal = self.lstm_gate(lstm_out, v)
        ctx = self.ctx_enrich(s).unsqueeze(1).expand(-1, x.shape[1], -1)
        enriched = self.enrich(temporal, ctx)
        att, _ = self.attn(enriched, enriched, enriched,
                           attn_mask=causal_mask(x.shape[1], x.device), need_weights=False)
        h = self.attn_gate(att, enriched)
        h = self.ffn(h)
        return self.out_gate(h, temporal)


class DeePMLite(DeepMomentumModel):
    """Temporal backbone + cross-sectional attention (portfolio-aware)."""

    cross_sectional = True

    def __init__(self, num_features, num_assets, hidden, dropout, heads=4, lagged_cross_section=False, **kw):
        super().__init__(num_features, num_assets, hidden, dropout)
        self.lagged = lagged_cross_section
        self.asset_emb = nn.Embedding(num_assets, hidden)
        self.vsn = VariableSelectionNetwork(num_features, hidden, dropout, context_size=hidden)
        self.init_h = nn.Linear(hidden, hidden)
        self.init_c = nn.Linear(hidden, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm_ffn = ResSwiGLU(hidden, dropout)
        self.temporal = AttentionBlock(hidden, heads, dropout)
        self.cross = AttentionBlock(hidden, heads, dropout, rezero_init=0.01)

    def encode(self, x, asset_ids):
        s = self.asset_emb(asset_ids)
        v, _ = self.vsn(x, s)
        h0 = torch.tanh(self.init_h(s)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(s)).unsqueeze(0)
        out, _ = self.lstm(v, (h0, c0))
        out = self.lstm_ffn(out)
        return self.temporal(out, attn_mask=causal_mask(x.shape[1], x.device))

    def cross_section(self, h, mask):
        B, N, L, H = h.shape
        q = h.permute(0, 2, 1, 3).reshape(B * L, N, H)
        if self.lagged:
            kv = torch.zeros_like(h)
            kv[:, :, 1:] = h[:, :, :-1]
            kv = kv.permute(0, 2, 1, 3).reshape(B * L, N, H)
        else:
            kv = q
        pad = (~mask.bool()).unsqueeze(1).expand(B, L, N).reshape(B * L, N)
        out = self.cross(q, kv, kv, key_padding_mask=pad)
        return out.reshape(B, L, N, H).permute(0, 2, 1, 3)


MODELS = {"lstm": LSTMDMN, "mom_trans": MomentumTransformer, "deepm": DeePMLite}


def build_model(kind: str, **kwargs) -> DeepMomentumModel:
    return MODELS[kind](**kwargs)
