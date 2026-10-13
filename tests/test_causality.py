"""No look-ahead: values at time t must not change when future data is appended."""

import numpy as np
import pandas as pd
import pytest

from tradinglab.config import PortfolioConfig
from tradinglab.data.store import build_panel
from tradinglab.features import deep_features
from tradinglab.portfolio.sizing import size_weights
from tradinglab.strategies.classic import classic_candidates


def _truncate(hourly, end):
    return {s: h[h.index < end] for s, h in hourly.items()}


def test_features_are_causal(hourly):
    full = build_panel(hourly, None, "1d", 3)
    cut = full.close.index[300]
    part = build_panel(_truncate(hourly, cut), None, "1d", 3)
    f_full, f_part = deep_features(full.close), deep_features(part.close)
    for name in f_full:
        a = f_full[name].loc[:cut].iloc[-50:]
        b = f_part[name].loc[:cut].iloc[-50:]
        pd.testing.assert_frame_equal(a, b, check_freq=False, obj=name)


@pytest.mark.parametrize("strategy", classic_candidates([]), ids=lambda s: s.name)
def test_classic_signals_and_sizing_are_causal(hourly, strategy):
    full = build_panel(hourly, None, "1d", 3)
    cut = full.close.index[320]
    part = build_panel(_truncate(hourly, cut), None, "1d", 3)
    assert part.close.index[-1] == cut
    s_full = strategy.signals(full).loc[:cut]
    s_part = strategy.signals(part)
    pd.testing.assert_frame_equal(s_full.iloc[-30:], s_part.iloc[-30:], check_freq=False)
    cfg = PortfolioConfig(vol_lookback=20)
    w_full = size_weights(strategy.signals(full), full, cfg, 365.0, "perp", 30, smoothing_span=5).loc[:cut]
    w_part = size_weights(s_part, part, cfg, 365.0, "perp", 30, smoothing_span=5)
    np.testing.assert_allclose(w_full.iloc[-30:].to_numpy(), w_part.iloc[-30:].to_numpy(), atol=1e-12)


def test_exec_price_is_after_bar_close(hourly):
    p = build_panel(hourly, None, "1d", 3)
    t = p.close.index[10]
    sym = p.symbols[0]
    # Bar labelled t closes at t; execution price is the open of the hour starting t+3h.
    assert p.close.loc[t, sym] == pytest.approx(hourly[sym]["close"].loc[t - pd.Timedelta(hours=1)])
    assert p.exec_price.loc[t, sym] == pytest.approx(hourly[sym]["open"].loc[t + pd.Timedelta(hours=3)])
