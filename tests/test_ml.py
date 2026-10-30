"""Daily LightGBM strategy: trains only on targets known before the cutoff; signals are causal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglab.config import load_config
from tradinglab.strategies.ml import GBMForecast
from tradinglab.strategies.registry import by_name


def _model(**kw):
    m = GBMForecast(horizon=5, min_history_bars=60, rounds=30, **kw)
    m.min_train_rows = 200        # the synthetic market has 3 coins
    return m


def test_training_uses_only_targets_known_before_the_cutoff(panel):
    pytest.importorskip("lightgbm")
    end = panel.close.index[400]
    a, b = _model(), _model()
    a.fit(panel, end)
    b.fit(panel.slice(None, end + pd.Timedelta(days=2)), end)     # the future is simply absent
    np.testing.assert_allclose(a.signals(panel).to_numpy(), b.signals(panel).to_numpy(), equal_nan=True)
    with pytest.raises(RuntimeError):
        _model().fit(panel, panel.close.index[61])                # nothing labelled yet


def test_signals_are_causal_and_long_only(panel, tmp_path):
    pytest.importorskip("lightgbm")
    m = _model()
    m.fit(panel, panel.close.index[400])
    full = m.signals(panel)
    t = panel.close.index[410]
    cut = m.signals(panel.slice(None, t))
    pd.testing.assert_frame_equal(full.loc[:t], cut, check_freq=False)
    assert (full.fillna(0.0) >= 0).all().all() and full.notna().any().any()
    last = m.signals(panel, start=t)
    pd.testing.assert_frame_equal(last, full.loc[t:], check_freq=False)
    m.save(tmp_path / "m")
    n = _model()
    n.load(tmp_path / "m")
    pd.testing.assert_frame_equal(n.signals(panel), full)


def test_registry_builds_gbm_strategies():
    cfg = load_config()
    s = by_name(cfg, "ml_gbm5_lf")
    assert isinstance(s, GBMForecast) and s.long_only and s.horizon == 5 and s.trainable
    assert not by_name(cfg, "ml_gbm21").long_only
