"""Live engine: broker accounting, backtest↔live parity, deferred fills, crash recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglab.backtest.engine import run_backtest
from tradinglab.config import CostConfig
from tradinglab.data.store import build_panel
from tradinglab.live.broker import PaperBroker
from tradinglab.live.engine import LiveEngine
from tradinglab.live.feed import LiveFeed
from tradinglab.live.state import StateStore
from tradinglab.portfolio.sizing import size_weights
from tradinglab.strategies.classic import TSMOM

H = pd.Timedelta(hours=1)


class ReplayFeed(LiveFeed):
    """Replays synthetic hourly data as if it were arriving live.

    ``publish_delay`` mimics the public archive: a UTC day only becomes visible
    once ``publish_delay`` has elapsed after its end.
    """

    def __init__(self, full: dict[str, pd.DataFrame], publish_delay: pd.Timedelta | None = None,
                 funding_8h: float = 0.0):
        super().__init__(md=None, symbols=list(full), history_start="2024-01-01", mode="archive", exchanges=[],
                         default_funding_8h=funding_8h)
        self.full = full
        self.publish_delay = publish_delay
        self.now: pd.Timestamp | None = None

    def known_until(self) -> pd.Timestamp:
        if self.publish_delay is None:
            return self.now.floor("h")
        return (self.now - self.publish_delay).floor("D")

    def refresh(self) -> None:
        self.mode = "archive"
        k = self.known_until()
        self.hourly = {s: h[h.index + H <= k] for s, h in self.full.items()}
        self.rates = {s: pd.Series(dtype="float64") for s in self.full}
        self._closes = None

    def fill_price(self, symbol, ts):
        hour = ts.floor("h")
        if hour + H > self.known_until():
            return None  # that minute is not published yet
        return float(self.full[symbol].loc[hour, "open"])


def _selection():
    return {"strategies": ["tsmom"], "alphas": {"tsmom": 1.0}, "candidates": ["tsmom", "breakout"],
            "deep_models": {}}


def _run_days(engine, feed, days, hour: int, minute: int = 0):
    for d in days:
        now = d + pd.Timedelta(hours=hour, minutes=minute)
        feed.now = now
        engine.cycle(now)


def test_broker_accounting(tmp_path):
    store = StateStore(tmp_path / "s.sqlite")
    broker = PaperBroker(store, CostConfig(fee_bps=10, slippage_bps=0))
    now = pd.Timestamp("2024-01-01", tz="UTC")
    store.ensure_account("a", "x", 1000.0, now)
    oid = broker.submit("a", "X", 2.0, now, now, 100.0, "test")
    broker.fill(store.pending_orders("a")[0], 100.0, now)
    assert store.positions("a") == {"X": 2.0}
    assert store.account("a")["cash"] == pytest.approx(1000 - 200 - 0.2)
    eq, gross, net = broker.equity("a", {"X": 110.0})
    assert eq == pytest.approx(1000 - 0.2 + 20) and gross == pytest.approx(220) and net == pytest.approx(220)
    broker.apply_funding("a", "X", now, 0.001, 110.0, estimated=False)   # long pays
    assert store.account("a")["cash"] == pytest.approx(1000 - 200 - 0.2 - 0.22)
    sid = broker.submit("a", "X", -5.0, now, now, 100.0, "flip short")
    broker.fill([o for o in store.pending_orders("a") if o["id"] == sid][0], 100.0, now)
    assert store.positions("a") == {"X": -3.0}
    assert oid != sid


@pytest.mark.parametrize("instrument,funding", [("perp", 0.0), ("spot", 0.0), ("perp", 0.0003)])
def test_live_engine_matches_backtest(tmp_path, cfg, hourly, instrument, funding):
    """Same data, same strategy, same costs: live paper equity == backtest equity."""
    cfg.market.exec_lag_hours = 0
    cfg.market.instrument = instrument
    cfg.live.fill_latency_seconds = 0
    cfg.live.shadow_all = False
    cfg.portfolio.min_trade_notional = 0.0
    feed = ReplayFeed(hourly, funding_8h=funding)
    store = StateStore(tmp_path / "lab.sqlite")
    cfg.paths.state_dir = str(tmp_path)
    engine = LiveEngine(cfg, feed, store, _selection())
    days = pd.date_range("2024-09-01", "2024-12-31", freq="D", tz="UTC")
    _run_days(engine, feed, days, hour=0)

    panel = build_panel(hourly, {s: pd.Series(dtype="float64") for s in hourly}, "1d", 0, funding)
    w = size_weights(TSMOM().signals(panel), panel, cfg.portfolio, 365.0, instrument, cfg.universe.min_history_bars)
    bt = run_backtest(w, panel, cfg.costs, band=cfg.portfolio.rebalance_band, risk=cfg.risk, start=days[0],
                      end=days[-1])
    bt_eq = cfg.portfolio.initial_capital * (1 + bt.returns).cumprod()
    bt_eq.index = bt_eq.index + pd.Timedelta(days=1)          # equity known at the next execution time

    live = store.equity_curve("main")["equity"]
    common = bt_eq.index.intersection(live.index)
    assert len(common) > 100
    rel = (live.loc[common] / bt_eq.loc[common] - 1).abs()
    # Funding accrues on drifting units at each 8h stamp live vs on start-of-day weights in the
    # backtest – a second-order difference.
    assert rel.max() < (2e-4 if funding == 0 else 1e-3), rel.describe()
    assert len(store.orders("main")) > 20


def test_archive_mode_defers_fills_without_lookahead(tmp_path, cfg, hourly):
    cfg.live.shadow_all = False
    feed = ReplayFeed(hourly, publish_delay=pd.Timedelta(hours=2, minutes=30))
    store = StateStore(tmp_path / "lab.sqlite")
    cfg.paths.state_dir = str(tmp_path)
    engine = LiveEngine(cfg, feed, store, _selection())
    day = pd.Timestamp("2024-10-01", tz="UTC")
    feed.now = day + pd.Timedelta(hours=11, minutes=40)
    rep = engine.cycle(feed.now)
    assert rep.horizon == day and rep.decisions == ["main"] and rep.orders > 0
    # Fill price not knowable yet → orders stay pending, no marks after decision time.
    assert len(store.pending_orders("main")) == rep.orders
    # Same day, later: still nothing new is published.
    feed.now = day + pd.Timedelta(hours=20)
    rep2 = engine.cycle(feed.now)
    assert rep2.fills == 0 and not rep2.decisions
    # Next day after publication: fills at the 11:41 price, marks through midnight, new decision.
    feed.now = day + pd.Timedelta(days=1, hours=3)
    rep3 = engine.cycle(feed.now)
    assert rep3.fills == rep.orders and rep3.decisions == ["main"]
    filled = store.orders("main")
    first = filled[filled["status"] == "filled"].iloc[-1]
    sym = first["symbol"]
    expected = hourly[sym].loc[day + pd.Timedelta(hours=11), "open"]
    side = np.sign(first["qty"])
    assert first["fill_price"] == pytest.approx(expected * (1 + side * cfg.costs.slippage_bps * 1e-4))
    eq = store.equity_curve("main")
    assert eq.index.max() == day + pd.Timedelta(days=1)


def test_engine_recovers_after_restart(tmp_path, cfg, hourly):
    cfg.market.exec_lag_hours = 0
    cfg.live.fill_latency_seconds = 0
    days = pd.date_range("2024-09-01", "2024-10-31", freq="D", tz="UTC")

    def run(split: bool, sub: str):
        d = tmp_path / sub
        d.mkdir()
        cfg.paths.state_dir = str(d)
        feed = ReplayFeed(hourly)
        store = StateStore(d / "lab.sqlite")
        engine = LiveEngine(cfg, feed, store, _selection())
        if not split:
            _run_days(engine, feed, days, 0)
        else:
            _run_days(engine, feed, days[:25], 0)
            store.close()  # "crash"
            store = StateStore(d / "lab.sqlite")
            engine = LiveEngine(cfg, ReplayFeed(hourly), store, _selection())
            _run_days(engine, engine.feed, days[25:], 0)
        return {a["name"]: store.equity_curve(a["name"])["equity"] for a in store.accounts()}

    a, b = run(False, "a"), run(True, "b")
    assert set(a) == set(b) == {"main", "shadow:tsmom", "shadow:breakout"}
    for k in a:
        pd.testing.assert_series_equal(a[k], b[k])
