"""The strategy under research (the only file the autoresearch loop edits).

fit(train, emb)            -> model. ``train``: rows whose 4-hour outcome is known (features k_*, m_*, labels y_*,
                              era, origin, symbol, coin); ``emb``: their Kronos hidden states (n x 512).
predict(model, test, emb)  -> one target weight per test row (no labels). The harness caps gross exposure at 1.

Idea: market-neutral book on Kronos's volatility forecast (k_sd_4, 6-day average; long calm, short jumpy) plus short-term reversal (last 24 h return, 1-day average; half weight); positions ease 10 % per 4 hours, carried across months.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------------ helpers
def xs_weights(test: pd.DataFrame, signal: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score of ``signal`` per origin, scaled to gross exposure 1 (market-neutral)."""
    s = pd.Series(signal, index=test.index)
    g = s.groupby(test["origin"])
    z = (s - g.transform("mean")) / (g.transform("std") + 1e-12)
    return (z / z.abs().groupby(test["origin"]).transform("sum")).fillna(0.0).to_numpy()


def tail(train: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    return train[train["origin"] > train["origin"].max() - pd.Timedelta(days=days)]


def smooth(model, test: pd.DataFrame, col: str, span: int) -> np.ndarray:
    """EWM of a feature per coin over past origins, the month's rows preceded by the training tail (causal)."""
    t = model["tail"]
    h = pd.concat([t[["origin", "symbol", col]], test[["origin", "symbol", col]]], ignore_index=True)
    is_test = np.r_[np.zeros(len(t), bool), np.ones(len(test), bool)]
    h = h.assign(is_test=is_test, pos=np.r_[np.full(len(t), -1), np.arange(len(test))]).sort_values(["symbol", "origin"], kind="stable")
    h["v"] = h.groupby("symbol")[col].transform(lambda x: x.ewm(span=span).mean())
    h = h[h["is_test"]].sort_values("pos")
    return h["v"].to_numpy()


def ease(test: pd.DataFrame, target: np.ndarray, step: float) -> np.ndarray:
    """Move each coin's weight ``step`` of the way toward its target at every origin (causal, starts flat)."""
    t = pd.DataFrame({"origin": test["origin"].to_numpy(), "symbol": test["symbol"].to_numpy(), "w": target})
    W = t.pivot(index="origin", columns="symbol", values="w").fillna(0.0).sort_index()
    cur, out = np.zeros(W.shape[1]), np.zeros(W.shape)
    for i, row in enumerate(W.to_numpy()):
        cur = cur + step * (row - cur)
        out[i] = cur
    E = pd.DataFrame(out, index=W.index, columns=W.columns).stack()
    return E.reindex(pd.MultiIndex.from_arrays([t["origin"], t["symbol"]])).to_numpy()


# ----------------------------------------------------------------------------------------------- strategy
STEP = 0.10
# (column, direction, span in 4-hour periods, take log, weight)
SIGNALS = [("k_sd_4", -1, 36, True, 1.0), ("m_ret_24", -1, 6, False, 0.5)]


def fit(train, emb):
    return {"tail": tail(train, 28)}


def zs(h: pd.DataFrame, x: pd.Series) -> pd.Series:
    g = x.groupby(h["origin"])
    return (x - g.transform("mean")) / (g.transform("std") + 1e-12)


def predict(model, test, emb):
    t = model["tail"]
    cols = list(dict.fromkeys(c for c, *_ in SIGNALS))
    h = pd.concat([t[["origin", "symbol", *cols]], test[["origin", "symbol", *cols]]], ignore_index=True)
    h["pos"] = np.r_[np.full(len(t), -1), np.arange(len(test))]
    h = h.sort_values(["symbol", "origin"], kind="stable")
    parts = []
    for c, sign, span, lg, wt in SIGNALS:
        v = h.groupby("symbol")[c].transform(lambda x: x.ewm(span=span).mean())
        parts.append((v if not lg else np.log(v), sign * wt))
    h = h.assign(**{f"_s{i}": v for i, (v, _) in enumerate(parts)}).sort_values(["origin", "symbol"], kind="stable").reset_index(drop=True)
    sig = sum(sw * zs(h, h[f"_s{i}"]) for i, (_, sw) in enumerate(parts))
    w = ease(h, xs_weights(h, sig.to_numpy()), STEP)
    keep = h["pos"].to_numpy() >= 0
    return w[keep][np.argsort(h["pos"].to_numpy()[keep])]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare import evaluate

    evaluate(fit, predict, final="--final" in sys.argv)
