"""HTML report generation smoke tests (synthetic data)."""

import pandas as pd

from tests.test_live import ReplayFeed, _run_days, _selection
from tradinglab.live.engine import LiveEngine
from tradinglab.live.state import StateStore
from tradinglab.report.html import live_report, research_report
from tradinglab.research.tournament import run_tournament
from tradinglab.strategies.classic import EqualWeightLong, MACDTrend, TSMOM


def test_live_report_renders(tmp_path, cfg, hourly):
    cfg.market.exec_lag_hours = 0
    cfg.paths.state_dir = str(tmp_path)
    feed = ReplayFeed(hourly)
    store = StateStore(tmp_path / "lab.sqlite")
    engine = LiveEngine(cfg, feed, store, _selection())
    _run_days(engine, feed, pd.date_range("2024-09-01", "2024-09-15", freq="D", tz="UTC"), 0)
    out = live_report(cfg, store, tmp_path / "live.html")
    html = out.read_text()
    assert "Main account P&amp;L" in html and "renderLines" in html and "shadow:tsmom" in html


def test_tournament_and_research_report(tmp_path, cfg, panel):
    cfg.selection.min_dsr = 0.0
    cfg.selection.min_psr = 0.0
    cfg.selection.min_oos_sharpe = -10
    report, results = run_tournament([EqualWeightLong(), TSMOM(), MACDTrend()], panel, cfg,
                                     cfg.walkforward.holdout_start)
    assert set(results) == {"basket_long", "tsmom", "macd"}
    assert "basket_long" not in report.live_choice          # benchmarks are never selected
    assert report.live_choice and set(report.live_alphas) == set(report.live_choice)
    assert abs(sum(report.live_alphas.values()) - 1) < 1e-9
    html = research_report(report, tmp_path / "r.html").read_text()
    assert "Holdout Sharpe" in html and "tsmom" in html
