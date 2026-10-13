"""Tabular forecasters for intraday returns (fit on stacked (bar, asset) rows)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class RidgeForecaster:
    """Standardised ridge regression – the linear baseline."""

    name = "ridge"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.mu = self.sd = self.coef = None
        self.intercept = 0.0

    def fit(self, X, y, Xv=None, yv=None, asset=None, asset_v=None):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        Z = (X - self.mu) / self.sd
        A = Z.T @ Z + self.alpha * len(Z) * 1e-3 * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ (y - y.mean()))
        self.intercept = float(y.mean())
        return self

    def predict(self, X, asset=None):
        return ((X - self.mu) / self.sd) @ self.coef + self.intercept

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "ridge.npz", mu=self.mu, sd=self.sd, coef=self.coef, intercept=self.intercept,
                 alpha=self.alpha)

    def load(self, path: Path) -> "RidgeForecaster":
        z = np.load(path / "ridge.npz")
        self.mu, self.sd, self.coef = z["mu"], z["sd"], z["coef"]
        self.intercept, self.alpha = float(z["intercept"]), float(z["alpha"])
        return self


class LGBMForecaster:
    """Gradient-boosted trees with early stopping on a time-ordered validation tail.

    Heavily regularised (large leaves, sub-sampling, shallow learning rate):
    intraday returns are mostly noise and trees overfit fast otherwise.
    """

    name = "lgbm"

    def __init__(self, threads: int = 4, seed: int = 0, **params):
        self.params = {
            "objective": "huber", "alpha": 1.5, "learning_rate": 0.03, "num_leaves": 31,
            "min_data_in_leaf": 2000, "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
            "lambda_l2": 10.0, "verbose": -1, "num_threads": threads, "seed": seed,
        }
        self.params.update(params)
        self.booster = None
        self.best_iteration = None

    def fit(self, X, y, Xv=None, yv=None, asset=None, asset_v=None):
        import lightgbm as lgb

        X = np.column_stack([X, asset]) if asset is not None else X
        cat = [X.shape[1] - 1] if asset is not None else "auto"
        train = lgb.Dataset(X, y, categorical_feature=cat, free_raw_data=True)
        valid_sets, callbacks = [], []
        if Xv is not None and len(Xv):
            Xv = np.column_stack([Xv, asset_v]) if asset_v is not None else Xv
            valid_sets = [lgb.Dataset(Xv, yv, categorical_feature=cat, reference=train)]
            callbacks = [lgb.early_stopping(50, verbose=False)]
        self.booster = lgb.train(self.params, train, num_boost_round=600, valid_sets=valid_sets,
                                 callbacks=callbacks)
        self.best_iteration = self.booster.best_iteration or self.booster.current_iteration()
        return self

    def predict(self, X, asset=None):
        X = np.column_stack([X, asset]) if asset is not None else X
        return self.booster.predict(X, num_iteration=self.best_iteration)

    def feature_importance(self) -> np.ndarray:
        return self.booster.feature_importance(importance_type="gain")

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "lgbm.txt").write_text(self.booster.model_to_string(num_iteration=self.best_iteration))
        (path / "lgbm.json").write_text(json.dumps({"params": self.params, "best_iteration": self.best_iteration}))

    def load(self, path: Path) -> "LGBMForecaster":
        import lightgbm as lgb

        self.booster = lgb.Booster(model_str=(path / "lgbm.txt").read_text())
        meta = json.loads((path / "lgbm.json").read_text())
        self.params, self.best_iteration = meta["params"], None
        return self


FORECASTERS = {"lgbm": LGBMForecaster, "ridge": RidgeForecaster}
