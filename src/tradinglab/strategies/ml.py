"""Gradient-boosted forecast of each coin's next-days return (daily bars, pooled across coins).

Features are the causal trend/momentum set the deep models use (volatility-normalised returns over
1 day–1 year, multi-scale MACD, price z-scores, volatility regime) plus each coin's cross-sectional rank
on its 5/21/63-day returns and the market's own 21/63-day trend. The target is the vol-normalised return
from the next execution to ``horizon`` days later. A model trained at ``train_end`` only sees rows whose
target was complete before ``train_end``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.config import utc
from tradinglab.data.store import Panel
from tradinglab.features import DEEP_FEATURES, deep_features, ewm_vol
from tradinglab.portfolio.sizing import history_mask
from tradinglab.strategies.base import Strategy

RANKED = ("r5", "r21", "r63")
PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 300,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
          "verbose": -1, "num_threads": 4, "seed": 0, "deterministic": True}


class GBMForecast(Strategy):
    """LightGBM forecast of the next days' return per coin; long-only takes positive forecasts only."""

    trainable = True
    smooth_signals = False     # signals are computed for the latest bar only in live trading
    min_train_rows = 1000

    def __init__(self, horizon: int = 5, long_only: bool = True, rounds: int = 300, min_history_bars: int = 300,
                 name: str | None = None):
        self.horizon = horizon
        self.long_only = long_only
        self.rounds = rounds
        self.min_history_bars = min_history_bars
        self.name = name or f"ml_gbm{horizon}{'_lf' if long_only else ''}"
        self.threads = 4
        self.booster = None
        self.scale: float | None = None
        self.train_end: pd.Timestamp | None = None

    def describe(self) -> str:
        side = "long when the forecast is positive, flat otherwise" if self.long_only else "long/short"
        return f"LightGBM forecast of each coin's next {self.horizon}-day return from trend features ({side})."

    # ------------------------------------------------------------ data
    def features(self, panel: Panel) -> tuple[np.ndarray, list[str]]:
        close = panel.close
        f = deep_features(close)
        names = list(DEEP_FEATURES)
        for k in RANKED:
            f[f"xs_{k}"] = f[k].rank(axis=1, pct=True) - 0.5
            names.append(f"xs_{k}")
        mkt = close.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
        level = (1 + mkt).cumprod()
        mvol = ewm_vol(mkt.to_frame()).iloc[:, 0]
        for h in (21, 63):
            m = (level.pct_change(h) / (mvol * np.sqrt(h))).clip(-5, 5)
            f[f"mkt_r{h}"] = pd.DataFrame(np.repeat(m.to_numpy()[:, None], close.shape[1], axis=1),
                                          index=close.index, columns=close.columns)
            names.append(f"mkt_r{h}")
        X = np.stack([f[n].to_numpy(dtype="float64") for n in names], axis=-1)       # [T, N, F]
        ok = np.isfinite(X).all(axis=-1) & history_mask(close, self.min_history_bars).to_numpy()
        X[~ok] = np.nan
        return X, names

    def target(self, panel: Panel) -> np.ndarray:
        h = self.horizon
        ex = panel.exec_price
        fwd = ex.shift(-h) / ex - 1.0
        vol = ewm_vol(panel.close.pct_change(fill_method=None))
        return (fwd / (vol * np.sqrt(h))).clip(-5, 5).to_numpy(dtype="float64")

    # ------------------------------------------------------------ model
    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        import lightgbm as lgb

        train_end = utc(train_end)
        X, _ = self.features(panel)
        y = self.target(panel)
        idx = panel.close.index
        # The target of row t is known h bars + the execution lag after t: keep rows complete before train_end.
        known = idx + pd.Timedelta(days=self.horizon + 1) <= train_end
        rows = np.flatnonzero(known)
        Xr, yr = X[rows].reshape(-1, X.shape[-1]), y[rows].reshape(-1)
        ok = np.isfinite(Xr).all(axis=1) & np.isfinite(yr)
        if ok.sum() < self.min_train_rows:
            raise RuntimeError(f"{self.name}: only {int(ok.sum())} training rows before {train_end}")
        params = {**PARAMS, "num_threads": self.threads}
        self.booster = lgb.train(params, lgb.Dataset(Xr[ok], yr[ok]), num_boost_round=self.rounds)
        self.scale = float(np.std(self.booster.predict(Xr[ok]))) or 1.0
        self.train_end = train_end

    def signals(self, panel: Panel, start=None, end=None) -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError(f"{self.name} has not been trained")
        idx = panel.close.index
        lo = 0 if start is None else int(np.searchsorted(idx, utc(start)))
        hi = len(idx) if end is None else int(np.searchsorted(idx, utc(end)))
        X, _ = self.features(panel)
        part = X[lo:hi]
        flat = part.reshape(-1, part.shape[-1])
        ok = np.isfinite(flat).all(axis=1)
        pred = np.full(len(flat), np.nan)
        if ok.any():
            pred[ok] = self.booster.predict(flat[ok])
        sig = np.clip(pred.reshape(part.shape[:2]) / (2.0 * self.scale), -1.0, 1.0)
        if self.long_only:
            sig = np.clip(sig, 0.0, None)
        return pd.DataFrame(sig, index=idx[lo:hi], columns=panel.close.columns)

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "gbm.txt").write_text(self.booster.model_to_string())
        (path / "meta.json").write_text(json.dumps({"name": self.name, "horizon": self.horizon,
                                                    "long_only": self.long_only, "scale": self.scale,
                                                    "train_end": str(self.train_end)}, indent=1))

    def load(self, path: Path) -> None:
        import lightgbm as lgb

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        self.booster = lgb.Booster(model_str=(path / "gbm.txt").read_text())
        self.scale = meta["scale"]
        self.train_end = utc(meta["train_end"]) if meta.get("train_end") else None
