"""Deep momentum models: shapes, bounds, causality, persistence (tiny CPU runs)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tradinglab.data.store import build_panel  # noqa: E402
from tradinglab.strategies.deep.models import MODELS, build_model  # noqa: E402
from tradinglab.strategies.deep.trainer import sharpe_objective, strategy_returns  # noqa: E402
from tradinglab.strategies.registry import make_deep  # noqa: E402


@pytest.mark.parametrize("kind", list(MODELS))
def test_model_output_shape_bounds_and_mask(kind):
    torch.manual_seed(0)
    m = build_model(kind, num_features=5, num_assets=4, hidden=8, dropout=0.1, heads=2)
    m.eval()
    x = torch.randn(3, 4, 10, 5)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1], [0, 0, 1, 0]], dtype=torch.float32)
    pos = m(x, mask)
    assert pos.shape == (3, 4, 10)
    assert (pos.abs() <= 1).all()
    assert (pos[mask == 0] == 0).all()


@pytest.mark.parametrize("kind", list(MODELS))
def test_model_is_causal_in_time(kind):
    """Changing inputs at step k must not change outputs before k."""
    torch.manual_seed(0)
    m = build_model(kind, num_features=5, num_assets=3, hidden=8, dropout=0.0, heads=2)
    m.eval()
    x = torch.randn(2, 3, 12, 5)
    mask = torch.ones(2, 3)
    a = m(x, mask)
    x2 = x.clone()
    x2[:, :, 8:] += 5.0
    b = m(x2, mask)
    torch.testing.assert_close(a[..., :8], b[..., :8])


def test_sharpe_objective_prefers_profitable_positions():
    y = torch.full((2, 3, 20), 0.01) + 0.001 * torch.randn(2, 3, 20)
    s = torch.ones_like(y)
    mask = torch.ones(2, 3)
    long_ = strategy_returns(torch.ones_like(y), y, s, mask, 0.0)
    short = strategy_returns(-torch.ones_like(y), y, s, mask, 0.0)
    for level in ("asset", "portfolio"):
        assert sharpe_objective(long_, mask, 5, level, 365) < sharpe_objective(short, mask, 5, level, 365)
    churn = torch.sign(torch.randn_like(y))
    costly = strategy_returns(churn, y, s, mask, 0.01)
    free = strategy_returns(churn, y, s, mask, 0.0)
    assert costly.sum() < free.sum()


@pytest.mark.parametrize("kind", ["lstm", "deepm"])
def test_deep_strategy_trains_predicts_causally_and_roundtrips(tmp_path, cfg, hourly, kind):
    cfg.deep.seq_len = 20
    cfg.deep.burn_in = 5
    cfg.deep.hidden = 8
    cfg.deep.heads = 2
    cfg.deep.max_epochs = 2
    cfg.deep.seeds = 1
    cfg.deep.threads = 1
    panel = build_panel(hourly, None, "1d", 3)
    strat = make_deep(kind, cfg)
    train_end = panel.close.index[330]
    strat.fit(panel, train_end)
    sig = strat.signals(panel, start=train_end)
    assert sig.notna().any().any()
    assert (sig.abs().max() <= 1).all()
    # Causality at inference: truncating future bars leaves past signals unchanged.
    cut = panel.close.index[360]
    part = panel.slice(None, cut)
    pd.testing.assert_frame_equal(sig.loc[:cut].iloc[-10:], strat.signals(part, start=train_end).iloc[-10:],
                                  check_freq=False)
    # Persistence round trip.
    strat.save(tmp_path / "m")
    other = make_deep(kind, cfg)
    other.load(tmp_path / "m")
    np.testing.assert_allclose(other.signals(panel, start=train_end).to_numpy(), sig.to_numpy(), equal_nan=True)


def test_live_engine_trades_saved_deep_model(tmp_path, cfg, hourly):
    from tests.test_live import ReplayFeed
    from tradinglab.live.engine import LiveEngine
    from tradinglab.live.state import StateStore

    cfg.deep.seq_len, cfg.deep.burn_in, cfg.deep.hidden, cfg.deep.heads = 20, 5, 8, 2
    cfg.deep.max_epochs, cfg.deep.seeds, cfg.deep.threads = 2, 1, 1
    cfg.market.exec_lag_hours = 0
    cfg.live.fill_latency_seconds = 0
    cfg.paths.state_dir = str(tmp_path)
    panel = build_panel(hourly, None, "1d", 0)
    strat = make_deep("mom_trans", cfg)
    strat.fit(panel, panel.close.index[300])
    strat.save(tmp_path / "models" / strat.name)
    selection = {"strategies": [strat.name, "tsmom"], "alphas": {strat.name: 0.5, "tsmom": 0.5},
                 "candidates": [strat.name, "tsmom"], "deep_models": {strat.name: str(tmp_path / "models" / strat.name)}}
    feed = ReplayFeed(hourly)
    store = StateStore(tmp_path / "lab.sqlite")
    engine = LiveEngine(cfg, feed, store, selection)
    for d in pd.date_range("2024-11-01", "2024-11-10", freq="D", tz="UTC"):
        feed.now = d
        rep = engine.cycle(d)
        assert not rep.warnings, rep.warnings
    assert {a["name"] for a in store.accounts()} == {"main", "shadow:dl_mom_trans", "shadow:tsmom"}
    assert len(store.orders("shadow:dl_mom_trans")) > 0
    assert len(store.equity_curve("main")) > 100
