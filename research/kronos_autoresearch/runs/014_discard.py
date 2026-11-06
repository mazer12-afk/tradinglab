"""The strategy under research (the only file the autoresearch loop edits).

fit(train, emb)            -> model. ``train``: rows whose 4-hour outcome is known (features k_*, m_*, labels y_*,
                              era, origin, symbol, coin); ``emb``: their Kronos hidden states (n x 512).
predict(model, test, emb)  -> one target weight per test row (no labels). The harness caps gross exposure at 1.

Idea: learn from Kronos - a ridge model on all of Kronos's path statistics plus 16 principal components of its hidden state, refitted monthly on all history, predicts the next 24 hours' relative return; market-neutral on its (12-day averaged) prediction; positions ease 5 % per 4 hours.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

SPAN, STEP, NPC = 72, 0.05, 16


def kfeat(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("k_")]


def fit(train, emb):
    tr = train.sort_values(["symbol", "origin"], kind="stable")
    # target: the next 24 hours' return relative to the other coins (needs the next 6 rows of the same coin)
    y24 = tr.groupby("symbol")["y_4"].transform(lambda x: x[::-1].rolling(6, min_periods=6).sum()[::-1])
    y24 = y24 - y24.groupby(tr["origin"]).transform("mean")
    ok = y24.notna().to_numpy()
    idx = tr.index.to_numpy()[ok]
    pca = PCA(NPC, random_state=0).fit(emb[idx][::3])
    X = np.column_stack([train.loc[idx, kfeat(train)].to_numpy(np.float64), pca.transform(emb[idx])])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = Ridge(alpha=1000.0).fit((X - mu) / sd, y24.to_numpy()[ok])
    t = tail(train, 28)
    return {"tail": t, "tail_emb": emb[t.index.to_numpy()], "pca": pca, "m": m, "mu": mu, "sd": sd}


def score(model, frame, e):
    X = np.column_stack([frame[kfeat(frame)].to_numpy(np.float64), model["pca"].transform(e)])
    return model["m"].predict((X - model["mu"]) / model["sd"])


def predict(model, test, emb):
    t = model["tail"]
    h = pd.concat([t[["origin", "symbol"]], test[["origin", "symbol"]]], ignore_index=True)
    h["p"] = np.r_[score(model, t, model["tail_emb"]), score(model, test, emb)]
    h["pos"] = np.r_[np.full(len(t), -1), np.arange(len(test))]
    h = h.sort_values(["symbol", "origin"], kind="stable")
    h["sig"] = h.groupby("symbol")["p"].transform(lambda x: x.ewm(span=SPAN).mean())
    h = h.sort_values(["origin", "symbol"], kind="stable").reset_index(drop=True)
    w = ease(h, xs_weights(h, h["sig"].to_numpy()), STEP)
    keep = h["pos"].to_numpy() >= 0
    return w[keep][np.argsort(h["pos"].to_numpy()[keep])]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare import evaluate

    evaluate(fit, predict, final="--final" in sys.argv)
