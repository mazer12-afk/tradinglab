"""The strategy under research (the only file the autoresearch loop edits).

fit(train, emb)            -> model. ``train``: rows whose 4-hour outcome is known (features k_*, m_*, labels y_*,
                              era, origin, symbol, coin); ``emb``: their Kronos hidden states (n x 512).
predict(model, test, emb)  -> one target weight per test row (no labels). The harness caps gross exposure at 1.

Idea: market-neutral on Kronos's volatility forecast (k_sd_4 averaged over 2 days): long the calm coins, short the jumpy ones; positions move 25 % of the way to target each 4 hours.
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
SPAN, STEP = 12, 0.25


def fit(train, emb):
    return {"tail": tail(train)}


def predict(model, test, emb):
    return ease(test, xs_weights(test, -smooth(model, test, "k_sd_4", SPAN)), STEP)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare import evaluate

    evaluate(fit, predict, final="--final" in sys.argv)
