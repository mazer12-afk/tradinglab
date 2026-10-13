import numpy as np
import pandas as pd
import pytest

from tradinglab.backtest.engine import run_backtest
from tradinglab.backtest.metrics import (
    deflated_sharpe,
    max_drawdown,
    newey_west_tstat,
    probabilistic_sharpe,
    sharpe,
)
from tradinglab.config import CostConfig, PortfolioConfig, RiskConfig
from tradinglab.data.store import Panel
from tradinglab.portfolio.risk import DrawdownGuard
from tradinglab.portfolio.sizing import size_weights


def test_sharpe_and_drawdown_known_values():
    r = pd.Series([0.01, -0.01] * 50 + [0.02])
    assert sharpe(r, 1) == pytest.approx(r.mean() / r.std())
    dd = max_drawdown(pd.Series([0.1, -0.5, 0.2]))
    assert dd == pytest.approx(-0.5)


def test_psr_dsr_behave():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.002, 0.01, 1000))
    noise = pd.Series(rng.normal(0.0, 0.01, 1000))
    assert probabilistic_sharpe(good) > 0.99
    assert 0.05 < probabilistic_sharpe(noise) < 0.95
    trials = [s.mean() / s.std() for s in [pd.Series(rng.normal(0, 0.01, 1000)) for _ in range(50)]]
    # Deflating against many trials can only lower the probability.
    assert deflated_sharpe(good, trials) <= probabilistic_sharpe(good) + 1e-12
    assert abs(newey_west_tstat(good)) > 3


def _toy_panel(prices: list[float], funding: float = 0.0) -> Panel:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D", tz="UTC")
    close = pd.DataFrame({"X": prices}, index=idx)
    return Panel(close, close.copy(), close * 0 + 1, close * 0 + funding, "1d", 0)


def test_backtest_constant_weight_compounds_exactly():
    prices = [100, 110, 99, 120, 120]
    p = _toy_panel(prices)
    w = pd.DataFrame({"X": 1.0}, index=p.close.index)
    res = run_backtest(w, p, CostConfig(0, 0))
    expected = np.array(prices[1:]) / np.array(prices[:-1]) - 1
    np.testing.assert_allclose(res.returns.to_numpy(), expected)
    assert res.equity.iloc[-1] == pytest.approx(prices[-1] / prices[0])


def test_backtest_costs_funding_and_band():
    p = _toy_panel([100, 100, 100, 100], funding=0.001)
    w = pd.DataFrame({"X": [1.0, 1.0, 0.0, 0.0]}, index=p.close.index)
    res = run_backtest(w, p, CostConfig(fee_bps=5, slippage_bps=5))
    # day 0: buy 1.0 (10bp) + funding 0.1%; day 1: hold, funding; day 2: sell 1.0 (10bp)
    np.testing.assert_allclose(res.returns.to_numpy(), [-0.002, -0.001, -0.001], atol=1e-5)
    # A band larger than the change suppresses the trade entirely.
    w2 = pd.DataFrame({"X": [0.5, 0.52, 0.49]}, index=p.close.index[:3])
    res2 = run_backtest(w2, p, CostConfig(0, 0), band=0.05)
    assert res2.turnover.iloc[1:].sum() == 0


def test_backtest_short_position_profits_when_price_falls():
    p = _toy_panel([100, 90, 81])
    w = pd.DataFrame({"X": -1.0}, index=p.close.index)
    res = run_backtest(w, p, CostConfig(0, 0))
    assert res.returns.iloc[0] == pytest.approx(0.10)


def test_sizing_respects_caps_and_spot(panel):
    cfg = PortfolioConfig(target_vol=5.0, max_gross=1.0, max_weight=0.4, vol_lookback=20)
    sig = pd.DataFrame(-1.0, index=panel.close.index, columns=panel.close.columns)
    w = size_weights(sig, panel, cfg, 365, "perp")
    assert (w.abs().sum(axis=1) <= 1.0 + 1e-9).all()
    assert (w.abs() <= 0.4 + 1e-9).all().all()
    w_spot = size_weights(sig, panel, cfg, 365, "spot")
    assert (w_spot >= 0).all().all() and w_spot.abs().sum().sum() == 0


def test_drawdown_guard_scales_halts_and_recovers():
    g = DrawdownGuard(RiskConfig(dd_derisk_start=0.1, dd_halt=0.2, dd_floor_scale=0.5), cooldown_bars=2)
    assert g.update(100) == 1.0
    assert g.update(95) == 1.0
    assert g.update(85) == pytest.approx(1 - 0.5 * 0.5)
    assert g.update(79) == 0.0          # halt
    assert g.update(80) == 0.0          # cooling down
    assert g.update(81) == 0.0          # last cooldown bar → peak reset to 81
    assert g.update(81) == 1.0
    assert g.halts == 1
