"""Event features are causal: appending later data never changes an earlier event or its features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglab.daytrade import events as E
from tradinglab.daytrade.sim import Minutes

SYMS = ["AAA", "BBB", "CCC"]


def market(days=12, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days * 1440, freq="1min", tz="UTC")
    close = {}
    for k, s in enumerate(SYMS):
        r = rng.standard_t(3, len(idx)) * 4e-4
        r[rng.integers(0, len(idx), 40)] += rng.choice([-1, 1], 40) * 0.02      # occasional crash/spike
        close[s] = 100 * (1 + k) * np.exp(np.cumsum(r))
    c = pd.DataFrame(close, index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    m = Minutes(o, np.maximum(o, c) * 1.0002, np.minimum(o, c) * 0.9998, c)
    grid = pd.date_range(idx[0], idx[-1], freq="5min")
    met = {s: pd.DataFrame({"sum_open_interest_value": 1e8 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(grid)))),
                            "sum_taker_long_short_vol_ratio": np.exp(rng.normal(0, 0.2, len(grid))),
                            "sum_toptrader_long_short_ratio": np.exp(rng.normal(0, 0.1, len(grid))),
                            "count_long_short_ratio": np.exp(rng.normal(0, 0.1, len(grid)))}, index=grid) for s in SYMS}
    fund = {s: pd.Series(1e-4, index=pd.date_range("2024-12-01", periods=150, freq="8h", tz="UTC")) for s in SYMS}
    return m, met, fund


def test_event_features_do_not_depend_on_later_data():
    m, met, fund = market()
    full = E.event_table(m, met, fund)
    cut = pd.Timestamp("2025-01-09", tz="UTC")
    part = E.event_table(m.slice(None, cut - pd.Timedelta(minutes=1)), {s: d[d.index < cut] for s, d in met.items()}, fund)
    assert len(full) > 20
    a = full[full["t"] < cut - pd.Timedelta(hours=5)].reset_index(drop=True)
    b = part[part["t"] < cut - pd.Timedelta(hours=5)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["t", "symbol"]], b[["t", "symbol"]])
    np.testing.assert_allclose(a[E.FEATURES + ["label"]].to_numpy(), b[E.FEATURES + ["label"]].to_numpy(),
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    late = part[part["t"] > cut - pd.Timedelta(hours=3)]
    assert late["label"].isna().all()                  # a 4-hour label is unknown until 4 hours have passed


def test_training_never_sees_a_label_from_the_month_it_trades():
    pytest.importorskip("lightgbm")
    m, met, fund = market(days=12)
    ev = E.event_table(m, met, fund)
    month = pd.Timestamp("2025-01-08", tz="UTC")
    a = E.train(ev, month)
    poisoned = ev.copy()
    poisoned.loc[poisoned["t"] >= month - pd.Timedelta(minutes=E.HORIZON + 10), "label"] = 99.0
    b = E.train(poisoned, month)
    later = ev[ev["t"] >= month]
    np.testing.assert_allclose(E.predict(a, later), E.predict(b, later))
