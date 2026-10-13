"""Bar-by-bar trading policies for the live intraday engine.

Each policy turns a model output at bar t into target portfolio weights, keeping
whatever state it needs (forecast history, discrete positions, frozen sizes)
in a JSON-serialisable dict so the engine can persist it between bars and
across restarts. ``tests/test_intraday.py`` checks that stepping a policy
through history reproduces the vectorised research functions exactly, so live
behaviour is what research evaluated.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradinglab.config import IntradayConfig
from tradinglab.intraday.data import IntradayData
from tradinglab.intraday.research import signal_to_weights


@dataclass
class PolicySpec:
    """One live paper-trading account = one model + one trading rule."""

    name: str                  # account suffix, e.g. "lgbm_1h"
    model: str                 # "lgbm" or "deep"
    horizon: int = 12          # forecast horizon in bars (lgbm)
    rule: str = "selective"    # "selective" (hysteresis) | "continuous" (deep positions)
    enter: float = 3.0
    exit: float = 0.0
    min_hold: int = 12
    norm_window: int = 288 * 30
    description: str = ""


def candidate_weights(signal_t: pd.Series, d: IntradayData, cfg: IntradayConfig) -> pd.Series:
    """Weights for signal row t, sized exactly like research (vol scaling + portfolio vol target)."""
    frame = pd.DataFrame(np.nan, index=d.index, columns=d.symbols)
    frame.iloc[-1] = signal_t.reindex(d.symbols).to_numpy()
    return signal_to_weights(frame, d, cfg).iloc[-1]


class SelectivePolicy:
    """Hysteresis entries on a normalised, horizon-averaged forecast; size frozen while held.

    Mirrors ``research.selective_weights``:
      s_t     = mean of the last H forecasts (per asset)
      score_t = s_t / mean(|s|) over the trailing ``norm_window`` bars (min 288)
      pos_t   = hysteresis(score_t; enter, exit, min_hold)
      w_t     = weight sized at entry, kept until the position changes.
    """

    def __init__(self, spec: PolicySpec, symbols: list[str], state: dict | None = None):
        self.spec = spec
        self.symbols = list(symbols)
        n = len(symbols)
        st = state or {}
        self.preds = deque(st.get("preds", []), maxlen=spec.horizon)               # list of [n] lists
        self.abs_s = deque(st.get("abs_s", []), maxlen=spec.norm_window)
        self.pos = np.array(st.get("pos", [0.0] * n), dtype=float)
        self.held = np.array(st.get("held", [0] * n), dtype=int)
        self.frozen = np.array(st.get("frozen", [0.0] * n), dtype=float)
        self.last_bar = st.get("last_bar")

    def state(self) -> dict:
        return {"preds": [list(map(float, p)) for p in self.preds],
                "abs_s": [list(map(float, a)) for a in self.abs_s],
                "pos": self.pos.tolist(), "held": self.held.tolist(), "frozen": self.frozen.tolist(),
                "last_bar": self.last_bar}

    def score(self, pred: np.ndarray) -> np.ndarray:
        """Update forecast history with bar t's prediction and return score_t."""
        self.preds.append(np.asarray(pred, dtype=float))
        P = np.array(self.preds)                                 # [≤H, n]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN column → NaN (no view)
            s = np.nanmean(np.where(np.isfinite(P), P, np.nan), axis=0)
        s = np.where(np.isfinite(P).any(axis=0), s, np.nan)
        self.abs_s.append(np.abs(s))
        A = np.array(self.abs_s)
        cnt = np.isfinite(A).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.nansum(np.where(np.isfinite(A), A, 0.0), axis=0) / cnt
        scale = np.where(cnt >= 288, scale, np.nan)
        return s / scale

    def step_positions(self, score: np.ndarray) -> np.ndarray:
        sp = self.spec
        old = self.pos.copy()
        for i, z in enumerate(score):
            p, held = self.pos[i], self.held[i]
            if not np.isfinite(z):
                p, held = 0.0, 0
            elif p == 0.0:
                if z > sp.enter:
                    p, held = 1.0, 0
                elif z < -sp.enter:
                    p, held = -1.0, 0
            else:
                held += 1
                if held >= sp.min_hold:
                    if p > 0 and z < sp.exit:
                        p, held = (-1.0 if z < -sp.enter else 0.0), 0
                    elif p < 0 and z > -sp.exit:
                        p, held = (1.0 if z > sp.enter else 0.0), 0
            self.pos[i], self.held[i] = p, held
        return old

    def update(self, pred: np.ndarray, bar: pd.Timestamp, d: IntradayData | None = None,
               cfg: IntradayConfig | None = None, size: bool = True) -> pd.Series | None:
        """Advance one bar. With ``size`` (and data), return target weights for bar t."""
        old = self.step_positions(self.score(pred))
        self.last_bar = str(bar)
        if not size:
            # Bars replayed without trading (warm-up / gaps): positions change, sizes are re-taken
            # at the next sized bar.
            changed = self.pos != old
            self.frozen[changed | (self.pos == 0)] = 0.0
            return None
        cand = candidate_weights(pd.Series(self.pos, index=self.symbols), d, cfg).to_numpy()
        for i in range(len(self.symbols)):
            if self.pos[i] == 0.0:
                self.frozen[i] = 0.0
            elif self.pos[i] != old[i] or self.frozen[i] == 0.0:
                self.frozen[i] = cand[i]
        return pd.Series(self.frozen.copy(), index=self.symbols)


class ContinuousPolicy:
    """Positions straight from an end-to-end trained network, sized like research."""

    def __init__(self, spec: PolicySpec, symbols: list[str], state: dict | None = None):
        self.spec = spec
        self.symbols = list(symbols)
        self.last_bar = (state or {}).get("last_bar")

    def state(self) -> dict:
        return {"last_bar": self.last_bar}

    def update(self, positions: np.ndarray, bar: pd.Timestamp, d: IntradayData | None = None,
               cfg: IntradayConfig | None = None, size: bool = True) -> pd.Series | None:
        self.last_bar = str(bar)
        if not size:
            return None
        return candidate_weights(pd.Series(positions, index=self.symbols), d, cfg)


def make_policy(spec: PolicySpec, symbols: list[str], state: dict | None = None):
    return (SelectivePolicy if spec.rule == "selective" else ContinuousPolicy)(spec, symbols, state)


DEFAULT_POLICIES = [
    PolicySpec("lgbm_1h", "lgbm", horizon=12, rule="selective", enter=3.0, exit=0.0, min_hold=12,
               description="LightGBM 1-hour forecast; trades only extreme forecasts (|z|>3), exits on sign flip."),
    PolicySpec("lgbm_4h", "lgbm", horizon=48, rule="selective", enter=2.5, exit=0.0, min_hold=24,
               description="LightGBM 4-hour forecast; enters at |z|>2.5, holds ≥2h, exits on sign flip."),
    PolicySpec("deep_lstm_attn", "deep", rule="continuous",
               description="LSTM + causal attention network trained end-to-end on net-of-cost Sharpe."),
]
