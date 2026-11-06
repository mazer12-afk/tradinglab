"""Cache Kronos-small's view of the 12 coins every 4 hours of one month (input for the autoresearch harness).

For each origin (00, 04, ..., 20 UTC) and coin with 488 complete hourly candles before it:
  emb       the model's final hidden state at the last candle (512 numbers, float16)
  entropy   entropy of its next-token (coarse) distribution
  close     32 sampled paths: log return of the close 1..4 hours ahead (float16, 32 x 4)
  high/low  mean over paths of the candle's high / low, log relative to the last close (4 each)
  vol       mean over paths of predicted volume relative to the context's mean volume (4)
Sampling: temperature 1.0, nucleus 0.9 (Kronos's defaults), seeded by the origin time.

    python scripts/kronos/cache.py --month 2025-07 --out cache/
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab import kronos as K
from tradinglab.config import load_config
from tradinglab.data.store import MarketData

CONTEXT, HORIZON, PATHS, EVERY = 488, 4, 32, 4
log = logging.getLogger("kronos-cache")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--out", default="cache")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import torch

    start = pd.Timestamp(a.month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    cfg = load_config()
    md = MarketData(cfg.paths.data_dir)
    syms = cfg.intraday.symbols
    c = {s: K.hourly(md, s, (start - pd.Timedelta(days=22)).tz_localize(None)) for s in syms}
    tok, mdl = K.load(threads=a.threads)
    smp = K.Sampler(tok, mdl, T=1.0, top_p=0.9)
    rows = {k: [] for k in ("origin", "symbol", "last", "emb", "entropy", "close", "high", "low", "vol")}
    t0 = time.time()
    origins = pd.date_range(start, end, freq=f"{EVERY}h", inclusive="left")
    for n, t in enumerate(origins):
        ctx, ok = [], []
        for s in syms:
            x = c[s][(c[s].index >= t - pd.Timedelta(hours=CONTEXT)) & (c[s].index < t)]
            if len(x) == CONTEXT and x.notna().all().all() and (x["close"] > 0).all():
                ctx.append(x)
                ok.append(s)
        if not ok:
            continue
        z = smp.forecast(ctx, HORIZON, PATHS, generator=torch.Generator().manual_seed(int(t.timestamp())))
        h = smp.state["hidden"].numpy()
        lp = torch.log_softmax(smp.state["s1_logits"], dim=-1)
        ent = (-(lp.exp() * lp).sum(dim=-1)).numpy()
        for i, s in enumerate(ok):
            last = float(ctx[i]["close"].iloc[-1])
            rows["origin"].append(t.value)
            rows["symbol"].append(s)
            rows["last"].append(last)
            rows["emb"].append(h[i].astype(np.float16))
            rows["entropy"].append(float(ent[i]))
            rows["close"].append(np.log(np.maximum(z[i, :, :, 3], 1e-12) / last).astype(np.float16))
            rows["high"].append(np.log(np.maximum(z[i, :, :, 1], 1e-12) / last).mean(axis=0).astype(np.float16))
            rows["low"].append(np.log(np.maximum(z[i, :, :, 2], 1e-12) / last).mean(axis=0).astype(np.float16))
            rows["vol"].append((z[i, :, :, 4] / max(float(ctx[i]["volume"].mean()), 1e-12)).mean(axis=0).astype(np.float16))
        if (n + 1) % 30 == 0:
            log.info("%s: %d/%d origins, %.1fs each", t, n + 1, len(origins), (time.time() - t0) / (n + 1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.month}.npz", **{k: np.asarray(v) for k, v in rows.items()})
    log.info("%s: %d rows in %.0fs", a.month, len(rows["origin"]), time.time() - t0)


if __name__ == "__main__":
    main()
