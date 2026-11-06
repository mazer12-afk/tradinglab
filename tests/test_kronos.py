"""The lab's cached Kronos sampler draws exactly what Kronos's own sampler draws.

Needs Kronos's code and weights (downloaded from GitHub and the Hugging Face hub), so it runs only with
KRONOS_TEST=1."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not os.environ.get("KRONOS_TEST"), reason="set KRONOS_TEST=1 (downloads Kronos)")


def _inputs(B=2, L=488, H=24, seed=0):
    from tradinglab import kronos as K

    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(0, 1, (B, L)), axis=1)
    o = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    hi, lo = np.maximum(o, close) + rng.uniform(0, 0.5, (B, L)), np.minimum(o, close) - rng.uniform(0, 0.5, (B, L))
    vol = rng.lognormal(0, 0.5, (B, L))
    x = np.stack([o, hi, lo, close, vol, vol * close], axis=-1)
    x = ((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-5)).astype(np.float32)
    ts = pd.date_range("2026-01-01", periods=L + H, freq="1h")
    return x, np.repeat(K.time_stamps(ts[:L])[None], B, 0), np.repeat(K.time_stamps(ts[L:])[None], B, 0)


@pytest.fixture(scope="module")
def kronos():
    from tradinglab import kronos as K

    return K.load(threads=4)


@torch.no_grad()
def test_shared_context_forward_equals_kronos(kronos):
    """Context prefilled once, then tokens appended one or several at a time for 3 paths: the same hidden states
    and logits as Kronos's full forward pass on each path."""
    from tradinglab import kronos as K

    tok, mdl = kronos
    x, xs, _ = _inputs(B=2)
    s1, s2 = tok.encode(torch.tensor(x), half=True)
    L0, S = 400, 3
    rng = np.random.default_rng(3)
    t1 = torch.tensor(rng.integers(0, 1024, (2 * S, x.shape[1] - L0)))          # a different future per path
    t2 = torch.tensor(rng.integers(0, 1024, (2 * S, x.shape[1] - L0)))
    rep = lambda t: t.repeat_interleave(S, dim=0)                              # noqa: E731
    f1, f2 = torch.cat([rep(s1[:, :L0]), t1], 1), torch.cat([rep(s2[:, :L0]), t2], 1)
    stamp = rep(torch.tensor(xs))
    ref_logits, ref_ctx = mdl.decode_s1(f1, f2, stamp)
    hc, ctx = K.prefill(mdl.transformer, mdl.embedding([s1[:, :L0], s2[:, :L0]]) + mdl.time_emb(torch.tensor(xs[:, :L0])))
    gen, outs, p = [None] * len(mdl.transformer), [], L0
    for n in (1, 5, 1, x.shape[1] - L0 - 7):
        e = mdl.embedding([f1[:, p:p + n], f2[:, p:p + n]]) + mdl.time_emb(stamp[:, p:p + n])
        outs.append(mdl.norm(K.extend(mdl.transformer, e, ctx, gen, p, S)))
        p += n
    mine = torch.cat([rep(mdl.norm(hc)), *outs], dim=1)
    np.testing.assert_allclose(mine.numpy(), ref_ctx.numpy(), atol=3e-4)
    np.testing.assert_allclose(mdl.head(mine).numpy(), ref_logits.numpy(), atol=3e-3)


def test_cached_sampler_draws_kronos_paths(kronos):
    from model.kronos import auto_regressive_inference

    from tradinglab import kronos as K

    tok, mdl = kronos
    x, xs, ys = _inputs()
    H, S = ys.shape[1], 6
    torch.manual_seed(11)
    ref = auto_regressive_inference(tok, mdl, torch.tensor(x), torch.tensor(xs), torch.tensor(ys), K.MAX_CONTEXT, H,
                                    5, 1.0, 0, 0.9, S)[:, -H:]          # Kronos averages its S paths
    torch.manual_seed(11)
    mine = K.Sampler(tok, mdl, T=1.0, top_p=0.9).sample_normalised(x, xs, ys, S)
    assert mine.shape == (x.shape[0], S, H, 6)
    np.testing.assert_allclose(mine.mean(axis=1), ref, atol=1e-3)
