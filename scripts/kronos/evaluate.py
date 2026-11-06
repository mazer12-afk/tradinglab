"""Kronos-small on the lab's 12 coins, every day after its weights were published: 24 hourly candles ahead.

Each UTC midnight from 1 Jul 2025 the model sees the previous 488 hourly candles of each coin (USD-M perpetuals,
OHLCV and quote volume) and samples 100 paths for the next 24 hours (temperature 1.0, nucleus 0.9: Kronos's
defaults). Kept per forecast: the sampled 24-hour returns, the p10/p50/p90 of the return at every hour ahead,
the forecast's typical hourly volatility, and what actually happened. Days whose context or next 24 hours have
missing candles are skipped. Checkpointed, so it resumes where it stopped.

    python scripts/kronos/evaluate.py [--start 2025-07-01] [--end 2026-11-05] [--paths 100] [--out reports/kronos]
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

CONTEXT, HORIZON = 488, 24
log = logging.getLogger("kronos-eval")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-07-01")
    ap.add_argument("--end", default="2026-11-05")
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--out", default="reports/kronos")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    part = out / "forecasts.partial.parquet"

    cfg = load_config()
    md = MarketData(cfg.paths.data_dir)
    syms = cfg.intraday.symbols
    start, end = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC")
    assert start > K.WEIGHTS_DATE, "forecasts must start after the weights were published"
    c = {s: K.hourly(md, s, (start - pd.Timedelta(days=25)).tz_localize(None)) for s in syms}
    tok, mdl = K.load(threads=a.threads)
    smp = K.Sampler(tok, mdl, T=1.0, top_p=0.9)
    import torch

    done = pd.read_parquet(part) if part.exists() else pd.DataFrame()
    have = set(pd.to_datetime(done["origin"], utc=True)) if len(done) else set()
    rows = [done] if len(done) else []
    origins = [t for t in pd.date_range(start, end - pd.Timedelta(hours=HORIZON), freq="D") if t not in have]
    log.info("%d origins to forecast (%d done)", len(origins), len(have))
    t0, new = time.time(), []
    for n, t in enumerate(origins):
        ctx, fut, ok = [], [], []
        for s in syms:
            h = c[s]
            x = h[(h.index >= t - pd.Timedelta(hours=CONTEXT)) & (h.index < t)]
            y = h[(h.index >= t) & (h.index < t + pd.Timedelta(hours=HORIZON))]
            if len(x) == CONTEXT and len(y) == HORIZON and x.notna().all().all() and y.notna().all().all():
                ctx.append(x)
                fut.append(y)
                ok.append(s)
        if not ok:
            continue
        gen = torch.Generator().manual_seed(int(t.timestamp()))
        z = smp.forecast(ctx, HORIZON, a.paths, generator=gen)              # (coins, paths, 24, 6)
        for i, s in enumerate(ok):
            last = float(ctx[i]["close"].iloc[-1])
            r = np.log(np.maximum(z[i, :, :, 3], 1e-12) / last)               # (paths, 24) log returns
            hr = np.diff(np.concatenate([np.zeros((a.paths, 1)), r], axis=1), axis=1)
            act = np.log(fut[i]["close"].to_numpy() / last)
            act_h = np.diff(np.concatenate([[0.0], act]))
            q = np.percentile(r, [10, 50, 90], axis=0)
            ctx_ret = np.diff(np.log(ctx[i]["close"].to_numpy()))
            new.append({
                "origin": t, "symbol": s, "last": last,
                "ret24_samples": r[:, -1].astype("float32"),
                "q10": q[0].astype("float32"), "q50": q[1].astype("float32"), "q90": q[2].astype("float32"),
                "mean24": float(r[:, -1].mean()),
                "pred_vol": float(np.median(hr.std(axis=1))),                    # typical hourly vol of a path
                "act": act.astype("float32"), "act_vol": float(act_h.std()),
                "act_high": float(np.log(fut[i]["high"].max() / last)), "act_low": float(np.log(fut[i]["low"].min() / last)),
                "ctx_ret24": float(np.log(last / ctx[i]["close"].iloc[-25])),       # baselines' inputs
                "ctx_vol24": float(ctx_ret[-24:].std()), "ctx_vol_ewm": float(pd.Series(ctx_ret).ewm(span=72).std().iloc[-1]),
            })
        if (n + 1) % 10 == 0 or n == len(origins) - 1:
            rows_all = pd.concat(rows + [pd.DataFrame(new)], ignore_index=True)
            rows_all.to_parquet(part)
            el = time.time() - t0
            log.info("%s done (%d/%d) %.1fs per day, ~%.0f min left", t.date(), n + 1, len(origins), el / (n + 1),
                     el / (n + 1) * (len(origins) - n - 1) / 60)
    final = pd.concat(rows + [pd.DataFrame(new)], ignore_index=True).sort_values(["origin", "symbol"])
    final.to_parquet(out / "forecasts.parquet")
    part.unlink(missing_ok=True)
    log.info("saved %d forecasts to %s", len(final), out / "forecasts.parquet")


if __name__ == "__main__":
    main()
