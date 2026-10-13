"""Strategy adapter for deep momentum models (walk-forward trainable, persistable)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.config import DeepConfig, utc
from tradinglab.data.store import Panel
from tradinglab.strategies.base import Strategy
from tradinglab.strategies.deep.trainer import DeepData, DeepTrainer, prepare_data

log = logging.getLogger(__name__)

DESCRIPTIONS = {
    "lstm": "LSTM Deep Momentum Network (Lim, Zohren & Roberts 2019), Sharpe loss net of costs.",
    "mom_trans": "Momentum Transformer (Wood et al. 2021): VSN + LSTM + causal attention, Sharpe loss.",
    "deepm": "DeePM-lite (after Wood et al. 2026): cross-sectional attention, robust soft-min Sharpe loss.",
}


class DeepStrategy(Strategy):
    trainable = True

    def __init__(self, kind: str, cfg: DeepConfig, cost_bps: float, bars_per_year: float,
                 asset_vol_target: float, vol_lookback: int, min_history_bars: int,
                 valid_fraction: float = 0.15):
        self.kind = kind
        self.name = f"dl_{kind}"
        self.cfg = cfg
        self.trainer = DeepTrainer(kind, cfg, cost_bps, bars_per_year)
        self.bpy = bars_per_year
        self.asset_vol_target = asset_vol_target
        self.vol_lookback = vol_lookback
        self.min_history_bars = min_history_bars
        self.valid_fraction = valid_fraction
        self.states: list[dict] = []
        self.symbols: list[str] | None = None
        self.train_end: pd.Timestamp | None = None

    def describe(self) -> str:
        return DESCRIPTIONS[self.kind]

    def _data(self, panel: Panel) -> DeepData:
        if self.symbols is not None and list(panel.close.columns) != self.symbols:
            panel = Panel(*(f.reindex(columns=self.symbols) for f in
                            (panel.close, panel.exec_price, panel.volume, panel.funding)),
                          panel.bar, panel.exec_lag_hours)
        return prepare_data(panel, self.asset_vol_target, self.vol_lookback, self.bpy, self.min_history_bars)

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        self.symbols = list(panel.close.columns)
        data = self._data(panel)
        self.states = self.trainer.fit(data, utc(train_end), valid_fraction=self.valid_fraction)
        self.train_end = utc(train_end)

    def signals(self, panel: Panel, start=None, end=None) -> pd.DataFrame:
        if not self.states:
            raise RuntimeError(f"{self.name} has not been trained")
        data = self._data(panel)
        idx = data.index
        lo = 0 if start is None else int(np.searchsorted(idx, utc(start)))
        hi = len(idx) if end is None else int(np.searchsorted(idx, utc(end)))
        ends = np.arange(max(lo, self.cfg.seq_len - 1), hi)
        pos = self.trainer.predict(data, self.states, ends)
        out = pd.DataFrame(np.nan, index=idx, columns=data.symbols)
        out.iloc[ends] = pos
        out = out.reindex(columns=panel.close.columns)
        if start is not None or end is not None:
            out = out.iloc[lo:hi]
        return out

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save([{"seed": s["seed"], "val_sharpe": s["val_sharpe"], "state": s["state"]}
                    for s in self.states], path / "weights.pt")
        meta = {"kind": self.kind, "symbols": self.symbols,
                "train_end": str(self.train_end) if self.train_end is not None else None,
                "val_sharpes": [s["val_sharpe"] for s in self.states],
                "deep_config": self.cfg.__dict__}
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    def load(self, path: Path) -> None:
        import torch

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        self.symbols = meta["symbols"]
        self.train_end = pd.Timestamp(meta["train_end"]) if meta["train_end"] else None
        self.states = torch.load(path / "weights.pt", weights_only=False)
