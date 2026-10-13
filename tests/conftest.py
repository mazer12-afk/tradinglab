"""Shared fixtures: deterministic synthetic markets (no network needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglab.config import load_config
from tradinglab.data.store import build_panel


def synthetic_hourly(symbols=("AAAUSDT", "BBBUSDT", "CCCUSDT"), days=420, seed=7,
                     start="2024-01-01", drift=0.0003, vol=0.01, trend_strength=0.15) -> dict[str, pd.DataFrame]:
    """Continuous hourly random walks with mild trending (open(h) == close(h-1))."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    out = {}
    for k, sym in enumerate(symbols):
        eps = rng.normal(0, vol, len(idx))
        # AR(1) on the drift component makes momentum strategies non-trivial.
        mu = np.zeros(len(idx))
        for t in range(1, len(idx)):
            mu[t] = (1 - 1 / 500) * mu[t - 1] + trend_strength * vol / 500 * rng.normal()
        r = drift / 24 + mu + eps / np.sqrt(24) * (1 + 0.3 * k)
        close = 100.0 * (1 + 0.5 * k) * np.exp(np.cumsum(r))
        open_ = np.concatenate([[close[0] / np.exp(r[0])], close[:-1]])
        high = np.maximum(open_, close) * 1.001
        low = np.minimum(open_, close) * 0.999
        vol_ = np.full(len(idx), 1000.0)
        out[sym] = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                                 "volume": vol_, "quote_volume": vol_ * close, "n": 1}, index=idx)
    return out


@pytest.fixture
def hourly():
    return synthetic_hourly()


@pytest.fixture
def panel(hourly):
    return build_panel(hourly, None, "1d", 3)


@pytest.fixture
def cfg(tmp_path):
    c = load_config(overrides={
        "universe": {"symbols": ["AAAUSDT", "BBBUSDT", "CCCUSDT"], "min_history_bars": 30},
        "walkforward": {"data_start": "2024-01-01", "first_test": "2024-06-01", "holdout_start": "2024-11-01",
                        "step_months": 3},
        "portfolio": {"vol_lookback": 20},
        "market": {"default_funding_8h": 0.0, "exec_lag_hours": 3},
    })
    c.paths.state_dir = str(tmp_path / "state")
    c.paths.models_dir = str(tmp_path / "models")
    c.paths.reports_dir = str(tmp_path / "reports")
    c.paths.data_dir = str(tmp_path / "data")
    return c
