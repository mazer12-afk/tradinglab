"""Strategy interface.

A strategy maps a market-data :class:`~tradinglab.data.store.Panel` to a frame of
*signals* in [-1, 1] (index = bar close time, columns = symbols). Signals are
converted into portfolio weights by a shared, causal sizing layer
(:mod:`tradinglab.portfolio.sizing`) so every strategy is risk-normalised the
same way in research and in live trading.

Contract: ``signals(panel).loc[t]`` may only depend on data with index <= t.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from tradinglab.data.store import Panel


class Strategy(ABC):
    #: unique identifier used in reports, the journal and the model registry
    name: str = "strategy"
    #: whether :meth:`fit` must be called (walk-forward retraining)
    trainable: bool = False
    #: signals are already portfolio weights (skip per-asset vol normalisation)
    outputs_weights: bool = False
    #: apply the portfolio-level EMA signal smoothing (rule-based strategies)
    smooth_signals: bool = True

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:  # pragma: no cover - default no-op
        """Train using only bars strictly before ``train_end``."""

    @abstractmethod
    def signals(self, panel: Panel) -> pd.DataFrame:
        """Signals in [-1, 1] for every bar in ``panel`` (NaN = no view)."""

    # Persistence hooks (trainable strategies override).
    def save(self, path: Path) -> None:  # pragma: no cover
        pass

    def load(self, path: Path) -> None:  # pragma: no cover
        pass

    def describe(self) -> str:
        return self.__doc__.strip().splitlines()[0] if self.__doc__ else self.name

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
