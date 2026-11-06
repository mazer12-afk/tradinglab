"""Kronos, the open-source foundation model for candlesticks (github.com/shiyu-coder/Kronos, MIT licence), in the lab.

Kronos turns OHLCV candles into discrete tokens and forecasts the next candles autoregressively; sampling many
paths gives a forecast distribution (the "fan" of paths and its p10-p90 band). Its own sampler recomputes the
whole 512-candle context for every path at every step. ``Sampler`` here gives the same draws with the context
computed once per series and shared by its paths (keys and values cached), so each new candle costs one
token's work per path; that makes a year of daily forecasts on 12 coins feasible on a CPU. The context must
leave room for the horizon inside Kronos's 512-token window (context + horizon - 1 <= 512), so no token drops
out of the window and the result is exact (checked against Kronos's own sampler in tests/test_kronos.py).

Weights: NeoQuasar/Kronos-small (24.7 M parameters) with NeoQuasar/Kronos-Tokenizer-base, both uploaded to the
Hugging Face hub on 30 Jun 2025 and unchanged since, so the model cannot have seen any later candle.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KRONOS_REPO = "https://github.com/shiyu-coder/Kronos.git"
KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"     # 13 Apr 2026, the code the evaluation used
MODEL = "NeoQuasar/Kronos-small"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
WEIGHTS_DATE = pd.Timestamp("2025-06-30", tz="UTC")             # nothing after this can be in the training data
MAX_CONTEXT = 512
COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def code_dir() -> Path:
    """Kronos's model code: $KRONOS_DIR, or a checkout of the pinned commit under data/ext/Kronos."""
    d = Path(os.environ.get("KRONOS_DIR", ROOT / "data" / "ext" / "Kronos"))
    if not (d / "model" / "kronos.py").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", KRONOS_REPO, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", KRONOS_COMMIT], check=True)
    return d


def load(model: str = MODEL, tokenizer: str = TOKENIZER, threads: int | None = None):
    """(tokenizer, model) in eval mode on the CPU."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    d = str(code_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    from model import Kronos, KronosTokenizer  # noqa: E402 - Kronos's own package

    return KronosTokenizer.from_pretrained(tokenizer).eval(), Kronos.from_pretrained(model).eval()


def time_stamps(ts) -> np.ndarray:
    """Kronos's calendar inputs: minute, hour, weekday, day, month."""
    ts = pd.DatetimeIndex(ts)
    return np.column_stack([ts.minute, ts.hour, ts.weekday, ts.day, ts.month]).astype(np.float32)


def _qkv(att, h, pos0: int):
    """Projected, RoPE-rotated queries and keys (and values) for tokens at positions pos0.. (as Kronos does)."""
    import torch

    B, n, _ = h.shape
    q, k, v = (p(h).view(B, n, att.n_heads, att.head_dim).transpose(1, 2) for p in (att.q_proj, att.k_proj, att.v_proj))
    t = torch.arange(pos0, pos0 + n, device=h.device).type_as(att.rotary.inv_freq)
    emb = torch.cat([torch.outer(t, att.rotary.inv_freq)] * 2, dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    rot = att.rotary._rotate_half
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin, v


def prefill(blocks, x):
    """A causal Kronos block stack over the shared context x (B, L, d): the output and each layer's (keys, values)."""
    import torch.nn.functional as F

    cache = []
    for blk in blocks:
        att = blk.self_attn
        q, k, v = _qkv(att, blk.norm1(x), 0)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + att.out_proj(o.transpose(1, 2).reshape(x.shape[0], x.shape[1], att.d_model))
        x = x + blk.ffn(blk.norm2(x))
        cache.append((k, v))
    return x, cache


def _attend(q, kc, vc, kg, vg, B: int, S: int, mask=None):
    """Attention of each path's queries q (B*S, h, n, d) over the context keys shared by the S paths of a series
    (kc, vc: (B, h, L, d)) and over the path's own keys (kg, vg: (B*S, h, g, d), optional)."""
    import torch

    BS, h, n, d = q.shape
    qs = q.view(B, S, h, n, d)
    parts = [torch.einsum("bshnd,bhkd->bshnk", qs, kc)]
    if kg is not None:
        sg = torch.einsum("bshnd,bshkd->bshnk", qs, kg.view(B, S, h, -1, d))
        parts.append(sg if mask is None else sg.masked_fill(~mask, float("-inf")))
    p = torch.softmax(torch.cat(parts, dim=-1) * d ** -0.5, dim=-1)
    L = kc.shape[2]
    o = torch.einsum("bshnk,bhkd->bshnd", p[..., :L], vc)
    if kg is not None:
        o = o + torch.einsum("bshnk,bshkd->bshnd", p[..., L:], vg.view(B, S, h, -1, d))
    return o.reshape(BS, h, n, d)


def extend(blocks, x, ctx: list, gen: list, pos0: int, S: int):
    """New tokens x (B*S, n, d) at positions pos0.. of each path through the block stack: attention over the
    shared context (``ctx`` from ``prefill``), the path's earlier tokens (``gen``, updated) and, causally, each
    other."""
    import torch

    BS, n, _ = x.shape
    B = BS // S
    for i, blk in enumerate(blocks):
        att = blk.self_attn
        q, k, v = _qkv(att, blk.norm1(x), pos0)
        if gen[i] is not None:
            k, v = torch.cat([gen[i][0], k], dim=2), torch.cat([gen[i][1], v], dim=2)
        gen[i] = (k, v)
        g = k.shape[2]
        mask = torch.ones(n, g, dtype=torch.bool, device=x.device).tril(g - n) if n > 1 else None
        o = _attend(q, *ctx[i], k, v, B, S, mask)
        x = x + att.out_proj(o.transpose(1, 2).reshape(BS, n, att.d_model))
        x = x + blk.ffn(blk.norm2(x))
    return x


class Sampler:
    """Kronos's autoregressive sampling (temperature, top-k / nucleus) with the context shared by all paths."""

    def __init__(self, tokenizer, model, T: float = 1.0, top_p: float = 0.9, top_k: int = 0, clip: float = 5.0):
        from model.kronos import top_k_top_p_filtering  # Kronos's own filter (its code is on sys.path after load)

        self.tok, self.mdl = tokenizer, model
        self.T, self.top_p, self.top_k, self.clip = T, top_p, top_k, clip
        self._filter = top_k_top_p_filtering

    def _draw(self, logits, generator=None):
        import torch
        import torch.nn.functional as F

        logits = logits / self.T
        if self.top_k > 0 or self.top_p < 1.0:
            logits = self._filter(logits, top_k=self.top_k, top_p=self.top_p)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=generator)

    def _s2_logits(self, a, ck, cv, gk, gv, last, B: int, S: int):
        """Kronos.decode_s2 at the newest position: the dependency layer's cross-attention from the drawn s1
        token over every earlier position (unmasked, and RoPE at position 0 is the identity), then the s2 head."""
        m = self.mdl
        ca = m.dep_layer.cross_attn
        q = ca.q_proj(m.embedding.emb_s1(a))
        q = q.view(q.shape[0], 1, ca.n_heads, ca.head_dim).transpose(1, 2)
        o = _attend(q, ck, cv, gk, gv, B, S)
        o = ca.out_proj(o.transpose(1, 2).reshape(q.shape[0], 1, ca.d_model))
        return m.head.cond_forward(m.dep_layer.norm(last + o))[:, -1]

    def _kv(self, h):
        ca = self.mdl.dep_layer.cross_attn
        B, n, _ = h.shape
        return tuple(p(h).view(B, n, ca.n_heads, ca.head_dim).transpose(1, 2) for p in (ca.k_proj, ca.v_proj))

    def sample_normalised(self, x, x_stamp, y_stamp, n_samples: int, generator=None) -> np.ndarray:
        """x: (B, L, 6) normalised candles, stamps (B, L, 5) and (B, H, 5) → (B, n_samples, H, 6) sampled paths."""
        import torch

        x = torch.as_tensor(x, dtype=torch.float32).clamp(-self.clip, self.clip)
        xs, ys = torch.as_tensor(x_stamp, dtype=torch.float32), torch.as_tensor(y_stamp, dtype=torch.float32)
        B, L, _ = x.shape
        H, S = ys.shape[1], n_samples
        if L + H - 1 > MAX_CONTEXT:
            raise ValueError(f"context {L} + horizon {H} must fit Kronos's {MAX_CONTEXT}-token window")
        m, tk = self.mdl, self.tok
        rep = lambda t: t.repeat_interleave(S, dim=0)                             # noqa: E731
        with torch.no_grad():
            s1, s2 = tk.encode(x, half=True)
            hc, ctx = prefill(m.transformer, m.embedding([s1, s2]) + m.time_emb(xs))
            hc = m.norm(hc)
            # the model's view of each series at the last context candle: its hidden state and next-token scores
            self.state = {"hidden": hc[:, -1].clone(), "s1_logits": m.head(hc[:, -1:])[:, -1].clone()}
            ck, cv = self._kv(hc)                                                  # the s2 layer's context keys
            logits, last = rep(m.head(hc[:, -1:])[:, -1]), rep(hc[:, -1:])
            gen, gk, gv, yst = [None] * len(m.transformer), None, None, rep(ys)
            pre, post = [], []
            for i in range(H):
                a = self._draw(logits, generator)
                b = self._draw(self._s2_logits(a, ck, cv, gk, gv, last, B, S), generator)
                pre.append(a)
                post.append(b)
                if i < H - 1:
                    e = m.embedding([a, b]) + m.time_emb(yst[:, i:i + 1])
                    last = m.norm(extend(m.transformer, e, ctx, gen, L + i, S))
                    logits = m.head(last)[:, -1]
                    nk, nv = self._kv(last)
                    gk = nk if gk is None else torch.cat([gk, nk], dim=2)
                    gv = nv if gv is None else torch.cat([gv, nv], dim=2)
            # the tokenizer's (causal) decoder over context + path, decoding only the path
            _, tctx = prefill(tk.decoder, tk.post_quant_embed(tk.indices_to_bits([s1, s2], half=True)))
            zg = tk.post_quant_embed(tk.indices_to_bits([torch.cat(pre, dim=1), torch.cat(post, dim=1)], half=True))
            z = tk.head(extend(tk.decoder, zg, tctx, [None] * len(tk.decoder), L, S))
        return z.reshape(B, S, H, -1).numpy()

    def forecast(self, candles: list[pd.DataFrame], horizon: int, n_samples: int, freq: str = "1h",
                 generator=None) -> np.ndarray:
        """Sampled future candles in price units for each context frame (index = candle open time, columns
        ``COLUMNS``; ``amount`` = quote volume). Returns (len(candles), n_samples, horizon, 6)."""
        xs, xst, yst, mu, sd = [], [], [], [], []
        for df in candles:
            v = df[COLUMNS].to_numpy(np.float32)
            m, s = v.mean(axis=0), v.std(axis=0)
            xs.append((v - m) / (s + 1e-5))
            mu.append(m)
            sd.append(s)
            xst.append(time_stamps(df.index))
            yst.append(time_stamps(pd.date_range(df.index[-1], periods=horizon + 1, freq=freq)[1:]))
        z = self.sample_normalised(np.stack(xs), np.stack(xst), np.stack(yst), n_samples, generator)
        return z * (np.stack(sd)[:, None, None, :] + 1e-5) + np.stack(mu)[:, None, None, :]


def hourly(md, symbol: str, start=None) -> pd.DataFrame:
    """1-hour USD-M perpetual candles from the archive, with Kronos's column names."""
    k = md.klines(symbol, "1h", start, market="um")
    return k.rename(columns={"quote_volume": "amount"})[COLUMNS].astype("float64")
