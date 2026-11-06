"""The strategy under research (the only file the autoresearch loop edits).

fit(train, emb)            -> model. ``train``: rows whose 4-hour outcome is known (features k_*, m_*, labels y_*,
                              era, origin, symbol, coin); ``emb``: their Kronos hidden states (n x 512).
predict(model, test, emb)  -> one target weight per test row (no labels). The harness caps gross exposure at 1.

Baseline: long or short each coin by the sign of Kronos's median 4-hour path, equal weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def fit(train, emb):
    return None


def predict(model, test, emb):
    return np.sign(test["k_med_4"].to_numpy()) / 12.0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare import evaluate

    evaluate(fit, predict, final="--final" in sys.argv)
