"""Intraday: causal features, policy live/research parity, real-time engine behaviour."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tradinglab.config import load_config
from tradinglab.data.exchanges import Quote
from tradinglab.intraday.data import IntradayData, assemble_intraday
from tradinglab.intraday.features import build_features
from tradinglab.intraday.policy import PolicySpec, SelectivePolicy
from tradinglab.intraday.research import selective_weights
from tradinglab.live.state import StateStore

SYMS = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]


def synthetic_5m(days=40, seed=1, start="2026-01-01") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 288, freq="5min", tz="UTC")
    out = {}
    for k, s in enumerate(SYMS):
        r = rng.normal(0, 0.002, len(idx)) + 0.0004 * np.sin(np.arange(len(idx)) / 50 + k)
        close = 100 * (1 + k) * np.exp(np.cumsum(r))
        open_ = np.concatenate([[close[0]], close[:-1]])
        qv = rng.uniform(1e5, 2e5, len(idx))
        out[s] = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.0005,
                               "low": np.minimum(open_, close) * 0.9995, "close": close, "volume": qv / close,
                               "quote_volume": qv, "taker_buy_volume": qv / close / 2,
                               "taker_buy_quote_volume": qv * rng.uniform(0.3, 0.7, len(idx)),
                               "trades": rng.integers(100, 200, len(idx)).astype(float)}, index=idx)
    return out


@pytest.fixture
def kl():
    return synthetic_5m()


@pytest.fixture
def icfg():
    cfg = load_config()
    cfg.intraday.symbols = SYMS
    return cfg


def test_intraday_features_are_causal(kl):
    d = assemble_intraday(kl, {}, {}, "5m", 0.0)
    cut = d.index[6000]
    d2 = assemble_intraday({s: k[k.index + pd.Timedelta(minutes=5) <= cut] for s, k in kl.items()}, {}, {}, "5m", 0.0)
    a, b = build_features(d, 12), build_features(d2, 12)
    assert a.names == b.names and "prem" not in a.names          # no perp basis on spot data
    np.testing.assert_allclose(a.X[5900:6001], b.X[5900:6001], rtol=1e-5, atol=1e-6, equal_nan=True)


def test_selective_policy_steps_like_research(kl, icfg):
    """Stepping the live policy bar by bar reproduces research.selective_weights exactly."""
    d = assemble_intraday(kl, {}, {}, "5m", 0.0)
    rng = np.random.default_rng(5)
    pred = pd.DataFrame(rng.normal(0, 1, (len(d.index), 3)), index=d.index, columns=SYMS)
    pred = pred.rolling(30, min_periods=1).mean()  # persistent forecasts → real entries/exits
    spec = PolicySpec("t", "lgbm", horizon=12, enter=1.5, exit=0.0, min_hold=6, norm_window=288 * 5)
    ic = icfg.intraday
    research = selective_weights(pred, d, ic, 12, 1.5, 0.0, 6, norm_window=spec.norm_window)
    pol = SelectivePolicy(spec, SYMS)
    check = range(len(d.index) - 120, len(d.index))
    first = check.start
    for t in range(first):
        pol.update(pred.iloc[t].to_numpy(), d.index[t], size=False)
    # Re-sync sizes: research sizes are taken at entries; start comparing once every open position
    # has been (re-)entered during the sized window.
    live = []
    for t in check:
        sub = d.slice(None, d.index[t])
        w = pol.update(pred.iloc[t].to_numpy(), d.index[t], sub, ic, size=True)
        live.append(w.to_numpy())
    live = np.array(live)
    res = research.iloc[list(check)].to_numpy()
    pos_live = np.sign(live)
    pos_res = np.sign(res)
    np.testing.assert_array_equal(pos_live, pos_res)             # identical entries/exits
    # Sizes agree wherever the position was opened inside the compared window.
    opened = np.zeros_like(pos_res, dtype=bool)
    for i in range(3):
        started = False
        for k in range(len(check)):
            if k > 0 and pos_res[k, i] != pos_res[k - 1, i]:
                started = True
            opened[k, i] = started and pos_res[k, i] != 0
    assert opened.any()
    np.testing.assert_allclose(live[opened], res[opened], rtol=1e-9, atol=1e-12)


def test_policy_state_roundtrip(kl, icfg):
    spec = PolicySpec("t", "lgbm", horizon=12, enter=1.0, exit=0.0, min_hold=2, norm_window=600)
    rng = np.random.default_rng(1)
    a = SelectivePolicy(spec, SYMS)
    for t in range(700):
        a.update(rng.normal(size=3), pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=5 * t), size=False)
    b = SelectivePolicy(spec, SYMS, json.loads(json.dumps(a.state())))
    x = rng.normal(size=3)
    assert np.allclose(a.score(x), b.score(x), equal_nan=True)
    assert (a.pos == b.pos).all() and (a.held == b.held).all()


class FakeExchange:
    """Serves the synthetic bars as if live, with a 2 bp spread around the last close."""

    name = "fake"

    def __init__(self, kl):
        self.full = kl
        self.now = None

    def klines(self, symbol, interval, start, full=False):
        k = self.full[symbol]
        return k[(k.index >= start) & (k.index + pd.Timedelta(minutes=5) <= self.now)]

    def quotes(self, symbols):
        out = {}
        for s in symbols:
            px = float(self.full[s]["close"][self.full[s].index + pd.Timedelta(minutes=5) <= self.now].iloc[-1])
            out[s] = Quote(px * 0.9999, px * 1.0001, self.now, "fake")
        return out


def test_realtime_engine_trades_and_accounts(tmp_path, kl, icfg, monkeypatch):
    lgb = pytest.importorskip("lightgbm")  # noqa: F841
    from tradinglab.intraday import live as L
    from tradinglab.intraday.models import LGBMForecaster
    from tradinglab.intraday.research import train_rows

    ic = icfg.intraday
    icfg.paths.state_dir = str(tmp_path)
    ic.train_days, ic.valid_days = 20, 3
    d = assemble_intraday(kl, {}, {}, "5m", 0.0)
    fs = build_features(d, 12)
    split = d.index[288 * 30]
    tr, va = train_rows(fs, ic, split)
    X, y, _, a = fs.rows(tr)
    m = LGBMForecaster(threads=1, min_data_in_leaf=50).fit(X, y, asset=a)
    mdir = tmp_path / "models"
    m.save(mdir / "lgbm_h12")
    (mdir / "manifest.json").write_text(json.dumps({"features": fs.names, "models": {"lgbm_h12": {}}}))

    ex = FakeExchange(kl)
    feed = L.IntradayFeed(md=None, symbols=SYMS, exchange=ex, buffer_days=25)
    monkeypatch.setattr(L, "okx_funding_rates", lambda syms, session=None: {s: 0.0001 for s in syms})
    ex.now = split
    feed.kl = {s: k[k.index + pd.Timedelta(minutes=5) <= split - pd.Timedelta(days=0)] for s, k in kl.items()}
    store = StateStore(tmp_path / "intraday.sqlite")
    spec = PolicySpec("lgbm_1h", "lgbm", horizon=12, enter=1.0, exit=0.0, min_hold=3, norm_window=288 * 5)
    eng = L.IntradayEngine(icfg, feed, store, mdir, policies=[spec])
    times = pd.date_range(split + pd.Timedelta(seconds=5), periods=288 * 3, freq="5min")
    traded = 0
    acct = "intraday:lgbm_1h"
    for t in times:
        ex.now = t
        before = (store.account(acct)["cash"], store.positions(acct)) if store.account(acct) else None
        rep = eng.cycle(t)
        assert rep.new_bars == 1, (t, rep)
        traded += rep.orders
        # no look-ahead: the engine never processes a bar that has not closed
        assert rep.bar <= t
    assert traded > 0
    eq = store.equity_curve(acct)
    assert len(eq) == len(times)
    # Accounting identity: the bar-t mark (taken before bar-t trades) equals the previous
    # post-trade cash plus funding paid at t plus positions valued at the bar-t close.
    cash_before, pos_before = before
    t_mark = eq.index[-1]
    px = d.close.loc[t_mark]
    fund = store.conn.execute("select coalesce(sum(amount),0) from funding where account=? and ts=?",
                              (acct, t_mark.isoformat())).fetchone()[0]
    assert eq["equity"].iloc[-1] == pytest.approx(cash_before + fund + sum(u * px[s] for s, u in pos_before.items()),
                                                  rel=1e-12)
    orders = store.orders(acct)
    assert (orders["status"] == "filled").all()
    buys = orders[orders["qty"] > 0]
    assert (buys["fill_price"] > buys["ref_price"] * 0.99).all()
    # Fees are exactly the taker fee on filled notional.
    np.testing.assert_allclose(orders["fee"], (orders["qty"].abs() * orders["fill_price"]) * ic.fee_bps * 1e-4)
    # Re-running a cycle for the same bar is a no-op (idempotent).
    rep = eng.cycle(times[-1])
    assert rep.new_bars == 0 and rep.orders == 0
