"""Simulator: on the same market it trades exactly like the production engines, and refuses in-sample runs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.test_intraday import SPECS, SYMS, _engine, synthetic_5m
from tests.test_live import ReplayFeed, _selection
from tradinglab import simulate as SIM
from tradinglab.config import load_config
from tradinglab.data.exchanges import Quote
from tradinglab.intraday.data import assemble_intraday
from tradinglab.intraday.policy import PolicySpec
from tradinglab.live.engine import LiveEngine
from tradinglab.live.state import StateStore

SPREAD = 1e-4   # the production test exchange quotes ± 1 bp


def _filled(store, name):
    o = store.orders(name, limit=1_000_000)
    return o[o["status"] == "filled"].sort_values("id").reset_index(drop=True)


def test_sim_intraday_matches_production_engine(tmp_path, monkeypatch):
    """Precomputed signals + a two-week sizing window + no restarts = the live engine, bar for bar."""
    kl = synthetic_5m()
    cfg = load_config()
    cfg.intraday.symbols = SYMS
    specs = SPECS + [PolicySpec("n", "lgbm", horizon=12, rule="continuous", transform="demean"),
                     PolicySpec("basket", "rule:basket", rule="continuous")]
    eng, ex, store, times = _engine(tmp_path, kl, cfg, monkeypatch, specs, "live")
    eng.feed.buffer_days = 1000          # features over the same history as the simulator's
    from tests.test_intraday import _restart
    eng = _restart(eng, ex, store, cfg, monkeypatch, specs, thresholds=[0.001])
    eng.feed.buffer_days = 1000
    for t in times:
        ex.now = t
        eng.cycle(t)
    eng.liquidate(times[-1], reason="simulation end")

    out = tmp_path / "sim"
    out.mkdir()
    data = assemble_intraday(kl, {}, {}, "5m", 0.0)
    res = SIM.run_intraday(cfg, times[0], times[-1], out, policies=specs, thresholds=[0.001],
                           spreads={s: SPREAD for s in SYMS}, md=None, funding={}, models_dir=eng.models_dir,
                           data=data, latency=pd.Timedelta(seconds=5))
    assert res["bars"] == len(times)
    sim = StateStore(out / "intraday.sqlite")
    names = [a.name for a in eng.accounts]
    assert sorted(a["name"] for a in sim.accounts()) == sorted(names)
    swept = 0
    for name in names:
        live_f, sim_f = _filled(store, name), _filled(sim, name)
        assert len(live_f) == len(sim_f) > 0, name
        assert (live_f[["symbol", "reason"]] == sim_f[["symbol", "reason"]]).all().all(), name
        np.testing.assert_allclose(sim_f["qty"], live_f["qty"], rtol=1e-6)
        np.testing.assert_allclose(sim_f["fill_price"], live_f["fill_price"], rtol=1e-12)
        e1, e2 = store.equity_curve(name)["equity"], sim.equity_curve(name)["equity"]
        np.testing.assert_allclose(e2.to_numpy(), e1.to_numpy(), rtol=1e-8)
        assert sim.vault_total(name) == pytest.approx(store.vault_total(name), rel=1e-8)
        assert sim.positions(name) == {}
        swept += len(sim.vault_events(name))
    assert swept > 0, "test needs at least one vault sweep"


def test_sim_intraday_window_and_quotes_have_no_lookahead(tmp_path):
    kl = synthetic_5m(days=20)
    data = assemble_intraday(kl, {}, {}, "5m", 0.0)
    clock = SIM.SimClock()
    feed = SIM.SimIntradayFeed(data, clock, lambda: {}, window_bars=100, first_window_bars=500)
    clock.now = data.index[3000] + pd.Timedelta(seconds=8)
    assert feed.last_common_close() == data.index[3000]
    d = feed.data()
    assert d.index[-1] == data.index[3000] and len(d.index) == 500       # first cycle: long window (warm-up)
    clock.now = data.index[3001] - pd.Timedelta(seconds=1)          # bar 3001 not closed yet
    assert feed.last_common_close() == data.index[3000]
    d = feed.data()
    assert d.index[-1] == data.index[3000] and len(d.index) == 100
    q = SIM.quote(100.0, "AAAUSDT", clock.now, {"AAAUSDT": 2e-4})
    assert q.bid == pytest.approx(99.98) and q.ask == pytest.approx(100.02) and q.mid == pytest.approx(100.0)


class OpenQuoteFeed(ReplayFeed):
    """Reference realtime feed: bars up to the clock, quotes at the open of the current hour ± 1 bp."""

    def refresh(self) -> None:
        super().refresh()
        self.mode = "realtime"

    def quote(self, symbol):
        px = float(self.full[symbol].loc[self.now.floor("h"), "open"])
        return Quote(px * (1 - SPREAD), px * (1 + SPREAD), self.now, "ref")


class LiveFeedAt(ReplayFeed):
    def __init__(self, full, now):
        super().__init__(full)
        self.now = now
        self.refresh()


class FakeMD:
    def __init__(self, hourly):
        self.h = hourly

    def hourly(self, symbol, start):
        return self.h[symbol]

    def funding_8h(self, symbol, start):
        return pd.Series(dtype="float64")


def test_sim_daily_matches_engine_with_reference_feed(tmp_path, cfg, hourly):
    cfg.live.vault_thresholds = [0.002]
    cfg.portfolio.min_trade_notional = 0.0
    start, end = pd.Timestamp("2024-09-01 00:10", tz="UTC"), pd.Timestamp("2024-09-22 23:50", tz="UTC")
    cfg.paths.state_dir = str(tmp_path / "ref")
    feed = OpenQuoteFeed(hourly)
    store = StateStore(tmp_path / "ref" / "lab.sqlite")
    ref = LiveEngine(cfg, feed, store, _selection())
    ref._export = lambda rep: None
    times = [start] + list(pd.date_range(start.floor("h") + pd.Timedelta(hours=1, seconds=30), end, freq="h"))
    for t in times + [end]:
        feed.now = t
        ref.cycle(t)
    ref.liquidate(end, reason="simulation end")

    out = tmp_path / "sim"
    cfg.paths.state_dir = str(out)
    res = SIM.run_daily(cfg, start, end, out, spreads={s: SPREAD for s in cfg.universe.symbols},
                        md=FakeMD(hourly), selection=_selection())
    assert res["cycles"] == len(times)
    sim = StateStore(out / "lab.sqlite")
    names = [s.name for s in ref.specs]
    assert "main+vault0.2" in names
    swept = 0
    for name in names:
        a, b = _filled(store, name), _filled(sim, name)
        assert len(a) == len(b) > 0, name
        np.testing.assert_allclose(b["fill_price"], a["fill_price"], rtol=1e-12)
        np.testing.assert_allclose(b["qty"], a["qty"], rtol=1e-12)
        pd.testing.assert_series_equal(sim.equity_curve(name)["equity"], store.equity_curve(name)["equity"],
                                       rtol=1e-12)
        swept += len(sim.vault_events(name))
        # decisions are taken once a day, 30 s after the bar + execution lag
        dec = pd.read_sql("select decided_at from decisions where account=?", sim.conn, params=(name,))
        t = pd.to_datetime(dec["decided_at"], utc=True, format="ISO8601")
        assert ((t - t.dt.floor("D")) == pd.Timedelta(hours=cfg.market.exec_lag_hours, seconds=30)).all()
    assert swept > 0, "test needs at least one vault sweep"


def test_daily_models_see_only_published_funding(cfg):
    idx = pd.date_range("2024-08-25", "2024-09-10", freq="8h", tz="UTC")
    rates = {"AAAUSDT": pd.Series(np.arange(len(idx)) * 1e-5 + 1e-5, index=idx)}
    hidx = pd.date_range("2024-08-25", periods=400, freq="h", tz="UTC")
    hourly = {"AAAUSDT": pd.DataFrame({"open": 1.0, "close": 1.0 + np.arange(len(hidx)) / 1000}, index=hidx)}
    clock = SIM.SimClock(pd.Timestamp("2024-09-05 12:00:30", tz="UTC"))
    feed = SIM.SimDailyFeed(hourly, rates, clock, {}, 0.0001)
    feed.refresh()
    assert feed.rates["AAAUSDT"].index.max() < pd.Timestamp("2024-09-01", tz="UTC")   # monthly archive
    ev = feed.funding_events("AAAUSDT", pd.Timestamp("2024-09-04", tz="UTC"), clock.now)
    assert [e[0] for e in ev] == list(idx[(idx > "2024-09-04") & (idx <= clock.now)])
    assert all(r == pytest.approx(rates["AAAUSDT"][t]) and not est for t, r, est in ev)   # charged as settled
    assert feed.hourly["AAAUSDT"].index[-1] + pd.Timedelta(hours=1) <= clock.now
    # the cached schedule (built once per run) gives the live feed's per-call answers, past the last print too
    cached = SIM.SimDailyFeed(hourly, rates, clock, {}, 0.0001,
                              span=(pd.Timestamp("2024-09-01", tz="UTC"), pd.Timestamp("2024-09-20", tz="UTC")))
    for a, b in [("2024-09-01", "2024-09-01 08:00"), ("2024-09-03 05:00", "2024-09-04 17:00"),
                 ("2024-09-09 20:00", "2024-09-12 01:00"), ("2024-09-15", "2024-09-15 01:00")]:
        a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        assert cached.funding_events("AAAUSDT", a, b) == feed.funding_events("AAAUSDT", a, b)
    # prices and the data horizon: the live feed's own methods on the same bars
    ref = LiveFeedAt(hourly, clock.now)
    assert feed.horizon() == ref.horizon()
    for ts in ("2024-09-05 10:59", "2024-09-05 11:00", "2024-09-05 12:00:30", "2024-09-09", "2024-08-01"):
        ts = pd.Timestamp(ts, tz="UTC")
        assert feed.price_at("AAAUSDT", ts) == ref.price_at("AAAUSDT", ts)


def test_in_sample_windows_are_refused(tmp_path, cfg):
    (tmp_path / "models" / "intraday" / "live").mkdir(parents=True)
    (tmp_path / "models" / "intraday" / "live" / "manifest.json").write_text(
        json.dumps({"data_end": "2026-10-13 20:45:00+00:00"}))
    (tmp_path / "models" / "selection.json").write_text(json.dumps({"data_end": "2026-10-13 00:00:00+00:00"}))
    cfg.paths.models_dir = str(tmp_path / "models")
    early = pd.Timestamp("2026-09-01", tz="UTC")
    with pytest.raises(SIM.InSampleError):
        SIM.check_out_of_sample(cfg, early, ["intraday"], allow=False)
    with pytest.raises(SIM.InSampleError):
        SIM.check_out_of_sample(cfg, early, ["daily"], allow=False)
    notes = SIM.check_out_of_sample(cfg, early, ["intraday", "daily"], allow=True)
    assert len(notes) == 2
    assert SIM.check_out_of_sample(cfg, pd.Timestamp("2026-10-15", tz="UTC"), ["intraday", "daily"], False) == []
    with pytest.raises(SIM.InSampleError):
        SIM.simulate(cfg, early, early + pd.Timedelta(days=1), tmp_path / "out", books=("daily",))


def test_walk_forward_retrains_on_past_data_only(tmp_path, monkeypatch):
    """Monthly steps: each model sees only bars closed by its step; before the switch nothing changes."""
    pytest.importorskip("lightgbm")
    kl = synthetic_5m(days=70)
    cfg = load_config()
    cfg.intraday.symbols = SYMS
    cfg.intraday.train_days, cfg.intraday.valid_days = 20, 3
    specs = [s for s in SPECS]
    data = assemble_intraday(kl, {}, {}, "5m", 0.0)
    start, end = pd.Timestamp("2026-02-27 00:00:05", tz="UTC"), pd.Timestamp("2026-03-02 12:00", tz="UTC")
    common = dict(policies=specs, thresholds=[], spreads={}, md=None, funding={}, data=data)
    wf = tmp_path / "wf"
    wf.mkdir()
    res = SIM.run_intraday(cfg, start, end, wf, retrain=True, **common)
    assert res["steps"] == [str(start), "2026-03-01 00:00:00+00:00"]
    for step in (start, pd.Timestamp("2026-03-01", tz="UTC")):
        man = json.loads((wf / "models" / "intraday" / f"{step:%Y%m%d}" / "manifest.json").read_text())
        assert pd.Timestamp(man["data_end"]) <= step < pd.Timestamp(man["data_end"]) + pd.Timedelta(minutes=10)
        for m in man["models"].values():
            assert pd.Timestamp(m["train_to_label_end"]) <= step
    # the same run without the March retrain is identical until the switch
    one = tmp_path / "one"
    one.mkdir()
    SIM.run_intraday(cfg, start, end, one, models_dir=wf / "models" / "intraday" / f"{start:%Y%m%d}", **common)
    a, b = StateStore(wf / "intraday.sqlite"), StateStore(one / "intraday.sqlite")
    for name in ("intraday:a", "intraday:b"):
        e1, e2 = a.equity_curve(name)["equity"], b.equity_curve(name)["equity"]
        before = e1.index < pd.Timestamp("2026-03-01", tz="UTC")
        pd.testing.assert_series_equal(e1[before], e2[before])
        assert e1.index.equals(e2.index)                   # every bar marked once, across the restart


def test_walk_forward_guard_and_roster(tmp_path, cfg, hourly):
    (tmp_path / "models" / "intraday" / "live").mkdir(parents=True)
    (tmp_path / "models" / "intraday" / "live" / "manifest.json").write_text(json.dumps({"data_end": "2026-10-13"}))
    (tmp_path / "models" / "selection.json").write_text(json.dumps({"data_end": "2026-10-13"}))
    cfg.paths.models_dir = str(tmp_path / "models")
    early = pd.Timestamp("2026-06-01", tz="UTC")
    notes = SIM.check_out_of_sample(cfg, early, ["intraday", "daily"], allow=False, retrain=True)
    assert len(notes) == 1 and "left out" in notes[0]
    cfg.live.vault_thresholds = [0.01]
    eng = SIM.SimLiveEngine(cfg, ReplayFeed(hourly), StateStore(tmp_path / "lab.sqlite"), _selection(),
                            drop=("main",))
    names = [s.name for s in eng.specs]
    assert not any(n.startswith("main") for n in names) and "shadow:tsmom+vault1" in names
    assert SIM.month_starts(early + pd.Timedelta(days=3), pd.Timestamp("2026-08-01", tz="UTC")) == [
        early + pd.Timedelta(days=3), pd.Timestamp("2026-07-01", tz="UTC")]
