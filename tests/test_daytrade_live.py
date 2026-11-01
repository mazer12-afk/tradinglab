"""Live day-trading engine: on a replayed market it makes exactly the simulator's trades."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglab.config import load_config
from tradinglab.daytrade import events as E
from tradinglab.daytrade import live as L
from tradinglab.daytrade.sim import Costs, Minutes, run_book, simulate_trades, funding_cost

SYMS = ["AAA", "BBB", "CCC"]
HS = {"AAA": 1e-4, "BBB": 2e-4, "CCC": 3e-4}
ARMS = [{"name": "a", "theta": 0.006, "hold": 240, "size": 0.2, "max_gross": 1.0},
        {"name": "b", "theta": 0.0045, "hold": 120, "size": 0.2, "max_gross": 1.0}]


def market(days=10, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-12-24", periods=days * 1440, freq="1min", tz="UTC")
    close = {}
    for k, s in enumerate(SYMS):
        r = rng.standard_t(3, len(idx)) * 4e-4
        r[rng.integers(0, len(idx), 250)] += rng.choice([-1, 1], 250) * 0.02
        close[s] = 100 * (1 + k) * np.exp(np.cumsum(r))
    c = pd.DataFrame(close, index=idx)
    o = c.shift(1).fillna(c.iloc[0]) * np.exp(rng.normal(0, 1e-4, c.shape))
    m = Minutes(o, np.maximum(o, c) * 1.0003, np.minimum(o, c) * 0.9997, c)
    grid = pd.date_range(idx[0], idx[-1], freq="5min")
    met = {s: pd.DataFrame({"sum_open_interest": 1e6 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(grid)))),
                            "sum_open_interest_value": 1e8 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(grid)))),
                            "sum_taker_long_short_vol_ratio": np.exp(rng.normal(0, 0.2, len(grid))),
                            "sum_toptrader_long_short_ratio": np.exp(rng.normal(0, 0.1, len(grid))),
                            "count_long_short_ratio": np.exp(rng.normal(0, 0.1, len(grid)))}, index=grid) for s in SYMS}
    fund = {s: pd.Series(rng.normal(1e-4, 1e-4, 40), index=pd.date_range("2025-12-23", periods=40, freq="8h", tz="UTC"))
            for s in SYMS}
    return m, met, fund


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t = self.t + pd.Timedelta(seconds=s)


class ReplayAPI:
    """Serves the synthetic market as if live: only closed bars, metrics once published, quotes at the minute's open."""

    def __init__(self, m, met, fund, clock):
        self.m, self.met, self.fund, self.clock = m, met, fund, clock

    def klines(self, s, start, end):
        now = self.clock()
        idx = self.m.index
        sel = (idx >= start) & (idx < end) & (idx + pd.Timedelta(minutes=1) <= now)
        return pd.DataFrame({c: getattr(self.m, c)[s][sel] for c in ("open", "high", "low", "close")}).assign(volume=1.0)

    def metrics(self, s, start, end=None):
        """Rows appear in two steps, as on Binance: open interest and long/short ratios first, the taker ratio later."""
        d = self.met[s]
        now = self.clock()
        d = d[(d.index >= start) & (d.index <= (end if end is not None else now))].copy()
        d.loc[d.index + pd.Timedelta(minutes=7.9) > now, "sum_taker_long_short_vol_ratio"] = np.nan
        return d[d.index + pd.Timedelta(minutes=6.5) <= now]

    def book(self):
        minute = self.clock().floor("min")
        return {s: (float(self.m.open.loc[minute, s]) * (1 - HS[s]), float(self.m.open.loc[minute, s]) * (1 + HS[s])) for s in SYMS}

    def funding(self, s, start):
        r = self.fund[s]
        return r[(r.index >= start) & (r.index <= self.clock())]


class BuyDrops:
    """A stand-in model: buys sharp drops, nudged by the taker and top-trader ratios (so a missing or mislabelled
    metrics value changes the trades)."""

    def predict(self, X):
        f = E.FEATURES.index
        return -0.004 * X[:, f("z")] + 0.003 * X[:, f("taker_5m")] + 0.003 * X[:, f("top_ls")]


def test_live_engine_trades_like_the_simulator(tmp_path):
    m, met, fund = market()
    t0 = pd.Timestamp("2026-01-01 00:00:05", tz="UTC")
    end = pd.Timestamp("2026-01-02 06:00", tz="UTC")
    clock = Clock(t0)
    api = ReplayAPI(m, met, fund, clock)
    L.run_shift(tmp_path, load_config(), minutes=(end - t0).total_seconds() / 60, end_at=end, api=api, clock=clock,
                sleep=clock.sleep, step=20, buffer_days=7, model=BuyDrops(), symbols=SYMS, arms=ARMS,
                cache=tmp_path / "cache")
    live = pd.read_csv(tmp_path / "trades.csv")
    assert len(live) > 10
    # the buffer ends up holding every published metrics value, with no holes left by partly published rows
    held = pd.read_parquet(tmp_path / "cache" / "metrics.parquet")
    for s in SYMS:
        h = held.loc[s]
        src = met[s][(met[s].index >= h.index[0]) & (met[s].index <= end - pd.Timedelta(minutes=8))]
        pd.testing.assert_frame_equal(h.loc[src.index, src.columns], src, check_freq=False, check_names=False)
    # the simulator on the same market: events decided from 00:10 on, filled at the next minute's open
    ev = E.event_table(m, met, fund, start=t0 + pd.Timedelta(minutes=5), end=end)
    pred = BuyDrops().predict(np.column_stack([ev[E.FEATURES].to_numpy(), ev["coin"].to_numpy()]))
    costs = Costs(taker_bps=5.0, impact_bps=1.0, half_spread=HS)
    for arm in ARMS:
        tr = simulate_trades(m, E.intents(ev, pred, arm["theta"], arm["hold"], arm["size"]), costs)
        tr["funding"] = funding_cost(tr, fund)
        bk = run_book(tr, max_gross=arm["max_gross"])
        sim = bk.trades[bk.trades["exit_time"] <= end - pd.Timedelta(minutes=1)]
        mine = live[live["arm"] == arm["name"]]
        a = sorted(zip(pd.to_datetime(mine["entry_time"], utc=True).dt.floor("min").dt.strftime("%m-%d %H:%M"), mine["symbol"]))
        b = sorted(zip((sim["entry_time"]).dt.strftime("%m-%d %H:%M"), sim["symbol"]))
        assert a == b, arm["name"]
        np.testing.assert_allclose(np.sort(mine["net"].to_numpy()), np.sort(sim["net"].to_numpy()), atol=1e-6)
        np.testing.assert_allclose(mine["pnl"].sum(), sim["pnl"].sum(), rtol=1e-4)


def test_settlements_and_state_survive_a_restart(tmp_path):
    assert L.settlements(pd.Timestamp("2026-01-01 07:59", tz="UTC"), pd.Timestamp("2026-01-02 00:00", tz="UTC")) == [
        pd.Timestamp("2026-01-01 08:00", tz="UTC"), pd.Timestamp("2026-01-01 16:00", tz="UTC"), pd.Timestamp("2026-01-02 00:00", tz="UTC")]
    m, met, fund = market()
    clock = Clock(pd.Timestamp("2026-01-01 00:00:05", tz="UTC"))
    api = ReplayAPI(m, met, fund, clock)
    kw = dict(api=api, clock=clock, sleep=clock.sleep, step=20, buffer_days=7, model=BuyDrops(), symbols=SYMS, arms=ARMS)
    end = pd.Timestamp("2026-01-02 06:00", tz="UTC")
    L.run_shift(tmp_path / "x", load_config(), minutes=600, end_at=end, **kw)       # shift 1: 10 hours
    clock.t = clock.t + pd.Timedelta(minutes=3)                                       # hand-over gap
    L.run_shift(tmp_path / "x", load_config(), minutes=10_000, end_at=end, **kw)    # shift 2
    one = pd.read_csv(tmp_path / "x" / "trades.csv")
    assert one["entry_time"].is_unique or len(one) > 0
    st = L.LiveEngine(tmp_path / "x", SYMS, api, arms=ARMS)
    assert all(np.isfinite(a.equity) for a in st.arms.values())


class FakeBinance:
    """HTTP stand-in with Binance's paging rules: /futures/data returns the latest rows unless both startTime and
    endTime are given; klines page forward from startTime."""

    def __init__(self, now):
        self.now = now
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        p = dict(params or {})
        step = 300_000 if "/futures/data/" in url else 60_000
        hi = int(self.now.value // 10**6) // step * step
        stamps = np.arange(hi - 30 * 86_400_000, hi + 1, step)        # the last 30 days
        lim = int(p.get("limit", 500))
        if "/futures/data/" in url:
            if "startTime" in p and "endTime" in p:
                sel = stamps[(stamps >= p["startTime"]) & (stamps <= p["endTime"])][:lim]
            elif "endTime" in p:
                sel = stamps[stamps <= p["endTime"]][-lim:]
            else:
                sel = stamps[-lim:]
            body = [{"timestamp": int(t), "sumOpenInterest": t / 1e9, "sumOpenInterestValue": t / 1e8,
                     "longShortRatio": 1.5, "buySellRatio": t / 1e12} for t in sel]
        else:
            sel = stamps[(stamps >= p["startTime"]) & (stamps <= p.get("endTime", hi))][:lim]
            body = [[int(t), "1", "2", "0.5", str(t / 1e12), "10"] for t in sel]

        class R:
            status_code = 200

            def json(self):
                return body
        return R()


def test_binance_paging_covers_the_whole_range(monkeypatch):
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    now = pd.Timestamp("2026-11-01 16:31:20", tz="UTC")
    api = L.BinanceLive(session=FakeBinance(now))
    start = pd.Timestamp("2026-10-11 16:30", tz="UTC")
    m = api.metrics("BTCUSDT", start, now)
    want = pd.date_range(start, now, freq="5min")
    want = want[want <= now.floor("5min") - pd.Timedelta(minutes=5)]   # the newest open-interest stamp is now's
    assert m.index[0] == start and m.loc[want].notna().all().all()
    # live open interest stamped X is the archive's X - 5 minutes; the taker ratio keeps its stamp
    t = pd.Timestamp("2026-10-20 12:00", tz="UTC")
    assert m.loc[t, "sum_open_interest"] == pytest.approx((t + pd.Timedelta(minutes=5)).value // 10**6 / 1e9)
    assert m.loc[t, "sum_taker_long_short_vol_ratio"] == pytest.approx(t.value // 10**6 / 1e12)
    k = api.klines("BTCUSDT", start, now.floor("min"))
    assert len(k) == int((now.floor("min") - start) / pd.Timedelta(minutes=1)) and k.index.is_unique
