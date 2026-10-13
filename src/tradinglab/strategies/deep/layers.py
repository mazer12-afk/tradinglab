"""Neural building blocks for the deep momentum models.

Adapted from the Temporal Fusion Transformer components used by Wood et al.
(Momentum Transformer, 2021) and the DeePM reference implementation
(https://github.com/kieranjwood/deepm, MIT License, Copyright (c) 2025 Kieran Wood).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, hidden_size: int | None = None, dropout: float | None = None):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return F.glu(self.fc(x), dim=-1)


class GateAddNorm(nn.Module):
    """GLU(x) + skip, then LayerNorm."""

    def __init__(self, input_size: int, hidden_size: int | None = None, dropout: float | None = None):
        super().__init__()
        hidden_size = hidden_size or input_size
        self.glu = GatedLinearUnit(input_size, hidden_size, dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, skip):
        return self.norm(self.glu(x) + skip)


class GatedResidualNetwork(nn.Module):
    """GRN with optional static context (TFT eq. 2-4)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1,
                 context_size: int | None = None):
        super().__init__()
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = GateAddNorm(hidden_size, output_size, dropout)

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip is not None else x
        h = self.fc1(x)
        if self.context is not None and context is not None:
            h = h + self.context(context)
        h = self.fc2(F.elu(h))
        return self.gate(h, residual)


class PerVariableGRN(nn.Module):
    """``F`` independent GRNs (H → H) evaluated in one batched einsum.

    Equivalent to a ModuleList of GRNs evaluated with batched matmuls. Dropout is
    applied once on the VSN output instead of inside every per-variable GRN,
    which is the dominant CPU cost for these memory-bound shapes.
    Input/output: [..., F, H].
    """

    def __init__(self, num_vars: int, hidden: int, dropout: float):
        super().__init__()
        def p(*shape, fan_in):
            return nn.Parameter(torch.empty(*shape).uniform_(-fan_in ** -0.5, fan_in ** -0.5))
        self.w1, self.b1 = p(num_vars, hidden, hidden, fan_in=hidden), p(num_vars, hidden, fan_in=hidden)
        self.w2, self.b2 = p(num_vars, hidden, hidden, fan_in=hidden), p(num_vars, hidden, fan_in=hidden)
        self.wg, self.bg = p(num_vars, hidden, 2 * hidden, fan_in=hidden), p(num_vars, 2 * hidden, fan_in=hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        lead = x.shape[:-2]
        nv, hd = x.shape[-2], x.shape[-1]
        z = x.reshape(-1, nv, hd).transpose(0, 1)                  # [F, M, H]
        h = torch.baddbmm(self.b1.unsqueeze(1), z, self.w1)
        h = torch.baddbmm(self.b2.unsqueeze(1), F.elu(h), self.w2)
        g = torch.baddbmm(self.bg.unsqueeze(1), h, self.wg)
        out = F.glu(g, dim=-1).transpose(0, 1).reshape(*lead, nv, hd)
        return self.norm(out + x)


class VariableSelectionNetwork(nn.Module):
    """Per-variable embedding + GRN, mixed by softmax weights conditioned on context.

    Input ``x``: [B, L, F]; ``context``: [B, H] or None. Output: [B, L, H].
    """

    def __init__(self, num_vars: int, hidden: int, dropout: float, context_size: int | None = None):
        super().__init__()
        self.num_vars = num_vars
        self.emb_w = nn.Parameter(torch.randn(num_vars, hidden) * 0.5)
        self.emb_b = nn.Parameter(torch.zeros(num_vars, hidden))
        self.var_grn = PerVariableGRN(num_vars, hidden, dropout)
        self.drop = nn.Dropout(dropout)
        self.weight_grn = GatedResidualNetwork(num_vars * hidden, hidden, num_vars, dropout, context_size)

    def forward(self, x, context=None):
        emb = x.unsqueeze(-1) * self.emb_w + self.emb_b               # [B, L, F, H]
        flat = emb.flatten(-2)                                          # [B, L, F*H]
        ctx = context.unsqueeze(1).expand(-1, x.shape[1], -1) if context is not None else None
        weights = torch.softmax(self.weight_grn(flat, ctx), dim=-1)     # [B, L, F]
        processed = self.var_grn(emb)                                   # [B, L, F, H]
        return self.drop((processed * weights.unsqueeze(-1)).sum(-2)), weights


class ResSwiGLU(nn.Module):
    """x = LayerNorm(x + Dropout(SwiGLU(x)))."""

    def __init__(self, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.w_in = nn.Linear(hidden, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        val, gate = self.w_in(x).chunk(2, dim=-1)
        return self.norm(x + self.drop(self.w_out(self.drop(val * F.silu(gate)))))


class AttentionBlock(nn.Module):
    """Post-norm attention block with optional ReZero gate on the attention branch."""

    def __init__(self, hidden: int, heads: int, dropout: float, rezero_init: float | None = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout)
        self.ffn = ResSwiGLU(hidden, dropout)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        if rezero_init is None:
            self.register_buffer("alpha", torch.tensor(1.0))
        else:
            self.alpha = nn.Parameter(torch.tensor(float(rezero_init)))

    def forward(self, x, key=None, value=None, attn_mask=None, key_padding_mask=None):
        k = x if key is None else key
        v = x if value is None else value
        out, _ = self.attn(x, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                           need_weights=False)
        out = torch.nan_to_num(out)  # rows with every key masked
        x = self.norm(x + self.alpha * self.drop(out))
        return self.ffn(x)


def causal_mask(length: int, device=None) -> torch.Tensor:
    """Boolean mask, True = position may NOT be attended (future)."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
