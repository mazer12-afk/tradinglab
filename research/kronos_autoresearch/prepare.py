"""Fixed harness for the Kronos autoresearch loop (the agent must not edit this file).

Data: one row per coin every 4 hours (00, 04, ..., 20 UTC), Jan 2024 - Oct 2026, with
  * Kronos-small's outputs at that moment (scripts/kronos/cache.py): summary features of 32 sampled 4-hour paths
    (columns k_*) and its 512-number hidden state (the ``emb`` array, row-aligned),
  * market features known at that moment (columns m_*): past returns, volatility, volume, funding, time,
  * labels, given to ``fit`` only (columns y_*): the next 1 and 4 hours' log return of the coin.
Eras: ``pre`` before 1 Jul 2025 (Kronos's weights date from 30 Jun 2025, so it may have seen these candles; use
for training only), ``dev`` Jul 2025 - Mar 2026 (scored), ``holdout`` Apr - Oct 2026 (locked: scored once, at the
end, with --final).

Evaluation (walk-forward by month): for each month, ``strategy.fit`` gets every row whose 4-hour outcome was
complete before the month began, then ``strategy.predict`` gives a target weight for each of the month's rows
(without labels). The book holds those weights for 4 hours, rescaled so gross exposure is at most 1, pays
COST_BPS on every change of weight and the funding of any settlement held through. Score: annualised Sharpe of
the 4-hourly net returns over the dev months (higher is better). Time budget: TIME_BUDGET seconds per run.
"""

from __future__ import annotations

import math
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "autoresearch"
CACHE_BRANCH = "experiment/kronos-cache"
DEV = ("2025-07-01", "2026-04-01")
HOLDOUT = ("2026-04-01", "2026-11-01")
COST_BPS = 6.0                 # per unit of weight changed: 5 bp taker fee + 1 bp slippage
TIME_BUDGET = 300              # seconds for one full evaluation (all monthly fits and predictions)
PERIODS_PER_YEAR = 6 * 365
LABELS = ["y_1", "y_4"]


# ----------------------------------------------------------------------------------------------- data
def fetch_cache() -> Path:
    """The cache branch's npz files, extracted under data/autoresearch/cache."""
    out = DATA / "cache"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin", CACHE_BRANCH], check=True)
    names = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "--name-only", f"origin/{CACHE_BRANCH}"],
                           check=True, capture_output=True, text=True).stdout.split()
    for n in names:
        if n.endswith(".npz") and not (out / n).exists():
            blob = subprocess.run(["git", "-C", str(ROOT), "show", f"origin/{CACHE_BRANCH}:{n}"], check=True,
                                  capture_output=True).stdout
            (out / n).write_bytes(blob)
    return out


def _kronos_features(z: dict) -> pd.DataFrame:
    c = z["close"].astype(np.float32)                         # (N, 32, 4) sampled log returns 1..4 h ahead
    f = {}
    for h in range(4):
        x = c[:, :, h]
        f[f"k_mean_{h + 1}"] = x.mean(axis=1)
        f[f"k_med_{h + 1}"] = np.median(x, axis=1)
        f[f"k_p10_{h + 1}"] = np.percentile(x, 10, axis=1)
        f[f"k_p90_{h + 1}"] = np.percentile(x, 90, axis=1)
        f[f"k_sd_{h + 1}"] = x.std(axis=1)
        f[f"k_up_{h + 1}"] = (x > 0).mean(axis=1)
        f[f"k_high_{h + 1}"] = z["high"][:, h].astype(np.float32)
        f[f"k_low_{h + 1}"] = z["low"][:, h].astype(np.float32)
        f[f"k_vol_{h + 1}"] = z["vol"][:, h].astype(np.float32)
    f["k_path_vol"] = np.diff(np.concatenate([np.zeros((len(c), 32, 1), np.float32), c], axis=2), axis=2).std(axis=2).mean(axis=1)
    f["k_entropy"] = z["entropy"].astype(np.float32)
    return pd.DataFrame(f)


def _market(md, syms, start, end) -> tuple[dict, dict]:
    from tradinglab import kronos as K

    hourly = {s: K.hourly(md, s, (start - pd.Timedelta(days=10)).tz_localize(None)) for s in syms}
    fund = {}
    for s in syms:
        r = md.funding_8h(s, (start - pd.Timedelta(days=10)).tz_localize(None))
        idx = pd.DatetimeIndex(r.index)
        r.index = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")).round("h")
        fund[s] = r[~r.index.duplicated(keep="last")]
    return hourly, fund


def build() -> None:
    """data/autoresearch/frame.parquet (features, labels, era) and emb.npy (row-aligned hidden states)."""
    from tradinglab.config import load_config
    from tradinglab.data.store import MarketData

    files = sorted(fetch_cache().glob("*.npz"))
    parts, embs = [], []
    for p in files:
        z = dict(np.load(p))
        if not len(z["origin"]):
            continue
        d = pd.DataFrame({"origin": pd.to_datetime(z["origin"], utc=True), "symbol": z["symbol"].astype(str),
                          "last": z["last"].astype(float)})
        parts.append(pd.concat([d, _kronos_features(z)], axis=1))
        embs.append(z["emb"].astype(np.float16))
    df = pd.concat(parts, ignore_index=True)
    emb = np.concatenate(embs)
    cfg = load_config()
    md = MarketData(cfg.paths.data_dir)
    syms = sorted(df["symbol"].unique())
    hourly, fund = _market(md, syms, df["origin"].min(), df["origin"].max())
    closes = pd.DataFrame({s: hourly[s]["close"] for s in syms})
    lc = np.log(closes)
    hr = lc.diff()
    mkt = hr.mean(axis=1)
    rows = []
    for s, g in df.groupby("symbol"):
        t = pd.DatetimeIndex(g["origin"])
        prev = t - pd.Timedelta(hours=1)                       # the last closed candle opened at t - 1 h
        L = lc[s]
        at = lambda dt: L.reindex(prev - pd.Timedelta(hours=dt)).to_numpy()            # noqa: E731
        now = at(0)
        h = hourly[s]
        vol = h["volume"]
        r = {"idx": g.index}
        for k in (1, 4, 12, 24, 72, 168):
            r[f"m_ret_{k}"] = now - at(k)
        r["m_vol_24"] = hr[s].rolling(24).std().reindex(prev).to_numpy()
        r["m_vol_168"] = hr[s].rolling(168).std().reindex(prev).to_numpy()
        r["m_volume_z"] = (np.log(vol.rolling(24).mean()) - np.log(vol.rolling(168).mean())).reindex(prev).to_numpy()
        r["m_range_24"] = (np.log(h["high"].rolling(24).max()) - np.log(h["low"].rolling(24).min())).reindex(prev).to_numpy()
        cm = mkt.cumsum()
        for k in (4, 24):
            r[f"m_mkt_{k}"] = cm.reindex(prev).to_numpy() - cm.reindex(prev - pd.Timedelta(hours=k)).to_numpy()
        fr = fund[s]
        ft, fc = fr.index.to_numpy(), np.concatenate([[0.0], np.cumsum(fr.to_numpy())])
        i_now = np.searchsorted(ft, t.to_numpy(), side="right")                # settlements at or before t
        r["m_funding"] = np.where(i_now > 0, fr.to_numpy()[np.maximum(i_now - 1, 0)], np.nan)     # last settled rate
        r["m_hour"] = t.hour.to_numpy()
        r["m_dow"] = t.dayofweek.to_numpy()
        # labels: log return over the next 1 and 4 hours; funding paid by a long over (t, t + 4 h]
        r["y_1"] = L.reindex(t).to_numpy() - now
        r["y_4"] = L.reindex(t + pd.Timedelta(hours=3)).to_numpy() - now
        r["fund_4"] = fc[np.searchsorted(ft, (t + pd.Timedelta(hours=4)).to_numpy(), side="right")] - fc[i_now]
        rows.append(pd.DataFrame(r).set_index("idx"))
    df = df.join(pd.concat(rows))
    df["coin"] = pd.Categorical(df["symbol"], categories=syms).codes
    df["era"] = np.where(df["origin"] < pd.Timestamp(DEV[0], tz="UTC"), "pre",
                         np.where(df["origin"] < pd.Timestamp(HOLDOUT[0], tz="UTC"), "dev", "holdout"))
    keep = df[["y_4", "m_ret_168", "m_vol_168"]].notna().all(axis=1).to_numpy()
    df, emb = df[keep].reset_index(drop=True), emb[keep]
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA / "frame.parquet")
    np.save(DATA / "emb.npy", emb)
    print(f"frame: {len(df)} rows, {df['origin'].min()} -> {df['origin'].max()}; by era {df['era'].value_counts().to_dict()}")


def load() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(DATA / "frame.parquet")
    df["origin"] = pd.to_datetime(df["origin"], utc=True)
    return df, np.load(DATA / "emb.npy", mmap_mode="r")


# ------------------------------------------------------------------------------------------ evaluation
class Budget(Exception):
    pass


def _alarm(signum, frame):
    raise Budget(f"over the {TIME_BUDGET}s time budget")


def book(df: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Net return of each 4-hour period for target weights ``w`` (one per row of ``df``)."""
    W = pd.DataFrame({"origin": df["origin"], "symbol": df["symbol"], "w": w}).pivot(index="origin", columns="symbol", values="w").fillna(0.0)
    gross = W.abs().sum(axis=1)
    W = W.div(np.maximum(gross, 1.0), axis=0)                  # gross exposure capped at 1
    R = df.pivot(index="origin", columns="symbol", values="y_4").reindex_like(W)
    F = df.pivot(index="origin", columns="symbol", values="fund_4").reindex_like(W).fillna(0.0)
    full = pd.date_range(W.index.min(), W.index.max(), freq="4h")
    W = W.reindex(full).fillna(0.0)
    R, F = R.reindex(full).fillna(0.0), F.reindex(full).fillna(0.0)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    return (W * (np.exp(R) - 1)).sum(axis=1) - (W * F).sum(axis=1) - turnover * COST_BPS * 1e-4, turnover


def stats(r: pd.Series, turnover: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    sd = r.std()
    return {"sharpe": float(r.mean() / sd * math.sqrt(PERIODS_PER_YEAR)) if sd > 0 else 0.0,
            "total_return": float(eq.iloc[-1] - 1), "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "turnover_per_day": float(turnover.mean() * 6), "periods": int(len(r)),
            "by_month": {str(k): round(float(v), 4) for k, v in (1 + r).groupby(r.index.strftime("%Y-%m")).prod().sub(1).items()}}


def evaluate(fit, predict, final: bool = False, verbose: bool = True) -> dict:
    """Walk-forward over the dev months (or, with ``final``, the holdout months); returns the stats."""
    df, emb = load()
    lo, hi = (pd.Timestamp(x, tz="UTC") for x in (HOLDOUT if final else DEV))
    months = pd.date_range(lo, hi, freq="MS", inclusive="left")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIME_BUDGET)
    t0 = time.time()
    try:
        weights = np.zeros(len(df))
        scored = np.zeros(len(df), dtype=bool)
        for m in months:
            nxt = m + pd.offsets.MonthBegin(1)
            tr = (df["origin"] + pd.Timedelta(hours=4) <= m).to_numpy()
            te = ((df["origin"] >= m) & (df["origin"] < nxt)).to_numpy()
            if not te.any():
                continue
            model = fit(df[tr].reset_index(drop=True), np.asarray(emb[tr], dtype=np.float32))
            test = df[te].drop(columns=LABELS + ["fund_4"]).reset_index(drop=True)
            w = np.asarray(predict(model, test, np.asarray(emb[te], dtype=np.float32)), dtype=float)
            if w.shape != (te.sum(),) or not np.isfinite(w).all():
                raise ValueError("predict must return one finite weight per test row")
            weights[te] = w
            scored |= te
    finally:
        signal.alarm(0)
    d = df[scored]
    r, to = book(d, weights[scored])
    out = stats(r, to)
    out["seconds"] = round(time.time() - t0, 1)
    if verbose:
        print("---")
        print(f"score:            {out['sharpe']:.4f}")
        print(f"total_return:     {out['total_return']:+.4f}")
        print(f"max_drawdown:     {out['max_drawdown']:.4f}")
        print(f"turnover_per_day: {out['turnover_per_day']:.3f}")
        print(f"seconds:          {out['seconds']}")
        print(f"by_month:         {out['by_month']}")
    return out


if __name__ == "__main__":
    build()
