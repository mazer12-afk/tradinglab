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
        self.latency = pd.Timedelta(0)  # quote time minus cycle start

    def klines(self, symbol, interval, start, full=False):
        k = self.full[symbol]
        return k[(k.index >= start) & (k.index + pd.Timedelta(minutes=5) <= self.now)]

    def quotes(self, symbols):
        out = {}
        for s in symbols:
            px = float(self.full[s]["close"][self.full[s].index + pd.Timedelta(minutes=5) <= self.now].iloc[-1])
            out[s] = Quote(px * 0.9999, px * 1.0001, self.now + self.latency, "fake")
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
    monkeypatch.setattr(L, "okx_funding_rates", lambda syms, session=None: {s: (0.0001, None) for s in syms})
    ex.now, ex.latency = split, pd.Timedelta(seconds=7)
    feed.kl = {s: k[k.index + pd.Timedelta(minutes=5) <= split - pd.Timedelta(days=0)] for s, k in kl.items()}
    store = StateStore(tmp_path / "intraday.sqlite")
    spec = PolicySpec("lgbm_1h", "lgbm", horizon=12, enter=1.0, exit=0.0, min_hold=3, norm_window=288 * 5)
    eng = L.IntradayEngine(icfg, feed, store, mdir, policies=[spec])
    times = pd.date_range(split + pd.Timedelta(seconds=5), periods=288 * 3, freq="5min")
    traded = 0
    acct = "intraday:lgbm_1h"
    held = {}  # bar -> units held through it (positions before the cycle that marks it)
    for t in times:
        ex.now = t
        before = (store.account(acct)["cash"], store.positions(acct)) if store.account(acct) else None
        held[t.floor("5min")] = dict(before[1]) if before else {}
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
    # Fills are stamped with the time of the quote they executed at, not the cycle start.
    lag = pd.to_datetime(orders["fill_ts"], utc=True) - pd.to_datetime(orders["submit_ts"], utc=True)
    assert (lag == pd.Timedelta(seconds=7)).all()
    buys = orders[orders["qty"] > 0]
    assert (buys["fill_price"] > buys["ref_price"] * 0.99).all()
    # Fees are exactly the taker fee on filled notional.
    np.testing.assert_allclose(orders["fee"], (orders["qty"].abs() * orders["fill_price"]) * ic.fee_bps * 1e-4)
    # Exported status carries the market/skill attribution; check it against ground truth recorded
    # during the run (positions actually held through each bar) rather than the module's own helpers.
    _check_engine_attribution(json.loads((tmp_path / "intraday_status.json").read_text())["accounts"][acct],
                              store, acct, eq, d.close, held, eng)
    # Re-running a cycle for the same bar is a no-op (idempotent).
    rep = eng.cycle(times[-1])
    assert rep.new_bars == 0 and rep.orders == 0


# ----------------------------------------------------------------- regression tests from the review
def _engine(tmp_path, kl, icfg, monkeypatch, specs, sub="e"):
    from tradinglab.intraday import live as L
    from tradinglab.intraday.models import LGBMForecaster
    from tradinglab.intraday.research import train_rows

    pytest.importorskip("lightgbm")
    root = tmp_path / sub
    root.mkdir()
    ic = icfg.intraday
    icfg.paths.state_dir = str(root)
    ic.train_days, ic.valid_days = 20, 3
    d = assemble_intraday(kl, {}, {}, "5m", 0.0)
    fs = build_features(d, 12)
    split = d.index[288 * 30]
    mdir = tmp_path / "models"
    if not (mdir / "manifest.json").exists():
        tr, _ = train_rows(fs, ic, split)
        X, y, _, a = fs.rows(tr)
        LGBMForecaster(threads=1, min_data_in_leaf=50).fit(X, y, asset=a).save(mdir / "lgbm_h12")
        (mdir / "manifest.json").write_text(json.dumps({"features": fs.names, "symbols": SYMS,
                                                         "models": {"lgbm_h12": {}}}))
    ex = FakeExchange(kl)
    feed = L.IntradayFeed(md=None, symbols=SYMS, exchange=ex, buffer_days=25)
    monkeypatch.setattr(L, "okx_funding_rates", lambda syms, session=None: {s: (0.0001, None) for s in syms})
    ex.now = split
    feed.kl = {s: k[k.index + pd.Timedelta(minutes=5) <= split] for s, k in kl.items()}
    store = StateStore(root / "intraday.sqlite")
    eng = L.IntradayEngine(icfg, feed, store, mdir, policies=specs)
    times = pd.date_range(split + pd.Timedelta(seconds=5), periods=288, freq="5min")
    return eng, ex, store, times


SPECS = [PolicySpec("a", "lgbm", horizon=12, enter=1.0, exit=0.0, min_hold=3, norm_window=288 * 5),
         PolicySpec("b", "lgbm", horizon=12, enter=1.5, exit=0.0, min_hold=6, norm_window=288 * 5)]


def _run(eng, ex, times, fail_at=None, fail_what=None):
    for i, t in enumerate(times):
        ex.now = t
        if i == fail_at:
            if fail_what == "quotes":
                orig = eng.feed.quotes
                eng.feed.quotes = lambda: (_ for _ in ()).throw(ConnectionError("quote outage"))
            else:
                orig = eng._outputs
                def boom(spec, *a, **k):
                    if spec.name == "b":
                        raise RuntimeError("model failure")
                    return orig(spec, *a, **k)
                eng._outputs = boom
            with pytest.raises((ConnectionError, RuntimeError)):
                eng.cycle(t)
            if fail_what == "quotes":
                eng.feed.quotes = orig
            else:
                eng._outputs = orig
        eng.cycle(t)  # the daemon's retry


@pytest.mark.parametrize("fail_what", ["model", "quotes"])
def test_failed_cycle_is_retried_atomically(tmp_path, kl, icfg, monkeypatch, fail_what):
    """A mid-cycle failure must neither double-apply a bar to policy state nor drop its trades."""
    clean, ex1, s1, times = _engine(tmp_path, kl, icfg, monkeypatch, SPECS, "clean")
    _run(clean, ex1, times)
    faulty, ex2, s2, _ = _engine(tmp_path, kl, icfg, monkeypatch, SPECS, "faulty")
    _run(faulty, ex2, times, fail_at=100, fail_what=fail_what)
    for acct in ("intraday:a", "intraday:b"):
        assert s1.get_meta(f"policy:{acct}") == s2.get_meta(f"policy:{acct}")
        pd.testing.assert_series_equal(s1.equity_curve(acct)["equity"], s2.equity_curve(acct)["equity"],
                                       rtol=1e-12)
        n1 = s1.conn.execute("select count(*) from decisions where account=?", (acct,)).fetchone()[0]
        n2 = s2.conn.execute("select count(*) from decisions where account=?", (acct,)).fetchone()[0]
        assert n1 == n2 == len(times)


def test_gap_bars_marked_at_last_price(tmp_path, kl, icfg, monkeypatch):
    """A symbol missing bars must be marked at its last price, not dropped from equity."""
    eng, ex, store, times = _engine(tmp_path, kl, icfg, monkeypatch, SPECS[:1])
    _run(eng, ex, times[:150])
    acct = "intraday:a"
    pos = store.positions(acct)
    assert pos, "test needs an open position"
    sym = next(iter(pos))
    # Remove the next 6 bars of that symbol from the exchange (an outage), while others continue.
    k = ex.full[sym]
    gap = (k.index > times[149]) & (k.index <= times[149] + pd.Timedelta(minutes=30))
    ex.full[sym] = k[~gap]
    cash_before, pos_before = store.account(acct)["cash"], dict(store.positions(acct))
    ex.now = times[150]
    eng.cycle(times[150])
    eq = store.equity_curve(acct)["equity"].iloc[-1]
    assert abs(eq - cash_before) > 1.0  # position still valued
    assert eq == pytest.approx(cash_before + sum(
        u * float(ex.full[s]["close"][ex.full[s].index + pd.Timedelta(minutes=5) <= eq_t].iloc[-1])
        for s, u in pos_before.items() for eq_t in [store.equity_curve(acct).index[-1]]), rel=1e-9)


def test_lagging_symbol_does_not_freeze_book(tmp_path, kl, icfg, monkeypatch):
    eng, ex, store, times = _engine(tmp_path, kl, icfg, monkeypatch, SPECS[:1])
    _run(eng, ex, times[:20])
    halted = "CCCUSDT"
    ex.full[halted] = ex.full[halted][ex.full[halted].index <= times[19]]
    reps = []
    for t in times[20:40]:
        ex.now = t
        reps.append(eng.cycle(t))
    assert all(r.new_bars == 1 for r in reps[5:])          # the book keeps moving
    assert any("lagging" in w for r in reps for w in r.warnings)


def test_forming_bar_is_not_accepted():
    from tradinglab.data.exchanges import BinanceSpot

    t0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    rows = [[int((t0 + pd.Timedelta(minutes=5 * i)).timestamp() * 1000), "1", "1", "1", "1", "1", 0, "1", 1, "1", "1"]
            for i in range(3)]
    # Asked at 00:14:59.5 – the 00:10 bar is still forming even if the response arrives later.
    df = BinanceSpot.parse_klines(rows, full=True, bar=pd.Timedelta(minutes=5),
                                  asof=t0 + pd.Timedelta(minutes=14, seconds=59.5))
    assert list(df.index) == [t0, t0 + pd.Timedelta(minutes=5)]


def test_engine_refuses_misaligned_model_symbols(tmp_path, kl, icfg, monkeypatch):
    eng, ex, store, times = _engine(tmp_path, kl, icfg, monkeypatch, SPECS[:1])
    from tradinglab.intraday import live as L

    feed = L.IntradayFeed(md=None, symbols=list(reversed(SYMS)), exchange=ex)
    with pytest.raises(RuntimeError, match="misaligned"):
        L.IntradayEngine(icfg, feed, store, tmp_path / "models", policies=SPECS[:1])


# ------------------------------------------------------------------ P&L attribution
def _prices(seed=4, n=600):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.003, (n, 3)), axis=0)), index=idx, columns=SYMS)


def _book(close, target, fee_bps=0.0):
    """Engine-style book: mark at close_t BEFORE trading, then trade to ``target(t)`` weights at close_t.

    Returns (equity marks, true start-of-bar exposure, fills)."""
    P = close.to_numpy()
    units, cash = np.zeros(P.shape[1]), 100_000.0
    eq, x, fills = [], [], []
    for t in range(len(P)):
        e = cash + units @ P[t]
        eq.append(e)
        x.append(units @ P[t - 1] if t else 0.0)        # held through bar t, valued at close_{t-1}
        new = np.asarray(target(t), float) * e / P[t]
        q = new - units
        fee = np.abs(q) * P[t] * fee_bps * 1e-4
        cash -= q @ P[t] + fee.sum()
        fills += [{"bar": close.index[t], "symbol": s, "qty": q[i], "fee": fee[i], "slippage_cost": 0.0}
                  for i, s in enumerate(close.columns) if q[i] != 0]
        units = new
    return (pd.Series(eq, index=close.index), pd.Series(x, index=close.index),
            pd.DataFrame(fills, columns=["bar", "symbol", "qty", "fee", "slippage_cost"]))


def _attr(close, eq, fills, funding=None):
    from tradinglab.intraday.attribution import attribute, bar_costs, basket_returns, held_units, start_exposure

    fund = funding if funding is not None else pd.DataFrame(columns=["ts", "amount"])
    x = start_exposure(held_units(fills, eq.index, list(close.columns)), close)
    return attribute(eq, x, basket_returns(close), bar_costs(fills, fund, eq.index)), x


@pytest.mark.parametrize("net", [0.4, -1.0, 1.0])
def test_attribution_passive_basket_is_all_passive(net):
    # The case the end-of-bar exposure got wrong: at net -1 a passive basket short showed
    # large fake timing and an equal and opposite fake selection.
    close = _prices()
    eq, x_true, fills = _book(close, lambda t: [net / 3] * 3)
    a, x = _attr(close, eq, fills)
    np.testing.assert_allclose(x.to_numpy(), x_true.to_numpy(), rtol=1e-12, atol=1e-6)
    assert a["beta"] == pytest.approx(net, abs=1e-9)
    assert a["avg_exposure"] == pytest.approx(net, abs=1e-12)
    assert a["passive_pnl"] == pytest.approx(a["pnl"], abs=1e-6)
    assert abs(a["timing_pnl"]) < 1e-6 and abs(a["selection_pnl"]) < 1e-6 and a["costs"] == 0.0


def test_attribution_timing_selection_and_costs():
    from tradinglab.intraday.attribution import basket_returns

    close = _prices(seed=5)
    m = basket_returns(close).fillna(0.0).to_numpy()
    # An (impossible) perfect timer: weights decided at t are held through bar t+1.
    eq, _, fills = _book(close, lambda t: [0.5 * np.sign(m[t + 1]) / 3] * 3 if t + 1 < len(m) else [0] * 3,
                         fee_bps=5)
    a, _ = _attr(close, eq, fills)
    paid = fills.loc[fills["bar"] < close.index[-1], "fee"].sum()   # last bar's trades are after the last mark
    assert a["costs"] == pytest.approx(paid) and a["fees"] == pytest.approx(paid)
    assert a["timing_pnl"] > 0.9 * (a["pnl"] + a["costs"])
    assert abs(a["passive_pnl"]) < 0.1 * a["timing_pnl"]
    assert abs(a["selection_pnl"]) < 1e-6
    assert a["passive_pnl"] + a["timing_pnl"] + a["selection_pnl"] - a["costs"] == pytest.approx(a["pnl"])
    # A one-asset book: what it earns beyond its basket-equivalent exposure is selection.
    eq1, x1, f1 = _book(close, lambda t: [0.5, 0.0, 0.0])
    a1, _ = _attr(close, eq1, f1)
    r = close.pct_change(fill_method=None)
    direct = (x1 * (r[SYMS[0]] - basket_returns(close))).iloc[1:].sum()
    assert a1["selection_pnl"] == pytest.approx(direct, rel=1e-9)
    assert abs(a1["timing_pnl"]) < 1e-6


def test_bar_costs_window_and_funding_stamps():
    from tradinglab.intraday.attribution import bar_costs

    marks = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    fills = pd.DataFrame({"bar": [marks[0], marks[2], marks[3]], "symbol": SYMS[0], "qty": 1.0,
                          "fee": [1.0, 2.0, 4.0], "slippage_cost": [0.5, 0.25, 8.0]})
    fund = pd.DataFrame({"ts": [marks[0], marks[1] - pd.Timedelta(minutes=1), marks[3]], "amount": [-3.0, -5.0, 7.0]})
    c = bar_costs(fills, fund, marks)
    # trades decided at a bar hit the NEXT mark; the last bar's trades are outside the window
    assert c["fees"].tolist() == [0.0, 1.0, 0.0, 2.0] and c["spread"].tolist() == [0.0, 0.5, 0.0, 0.25]
    # funding settled at a stamp is charged at the first mark at or after it (positive = paid)
    assert c["funding"].tolist() == [3.0, 5.0, 0.0, -7.0]


def test_basket_returns_carry_last_price_over_gaps():
    from tradinglab.intraday.attribution import basket_returns

    close = _prices(n=200)
    gap = close.copy()
    gap.iloc[100:110, 0] = np.nan
    b = basket_returns(gap)
    r = close.pct_change(fill_method=None)
    assert b.iloc[105] == pytest.approx((0.0 + r.iloc[105, 1] + r.iloc[105, 2]) / 3)
    jump = close.iloc[110, 0] / close.iloc[99, 0] - 1
    assert b.iloc[110] == pytest.approx((jump + r.iloc[110, 1] + r.iloc[110, 2]) / 3)


def _check_engine_attribution(st, store, acct, eq, close, held, eng):
    from tradinglab.intraday.attribution import basket_returns

    at = st["attribution"]
    assert at is not None and at["window_start"] == str(eq.index[0]) and at["bars"] == len(eq) - 1
    assert at["pnl"] == pytest.approx(st["pnl"], rel=1e-9)
    # ground truth: units recorded before each cycle, valued at the previous bar's close
    prev = close.ffill().shift(1)
    x_true = pd.Series([sum(u * prev.loc[t, s] for s, u in held.get(t, {}).items()) for t in eq.index],
                       index=eq.index)
    m = basket_returns(close).reindex(eq.index)
    w = (x_true / eq["equity"].shift(1)).iloc[1:]
    unit = (m * eq["equity"].shift(1)).iloc[1:]
    assert at["passive_pnl"] == pytest.approx(w.mean() * unit.sum(), rel=1e-9, abs=1e-6)
    assert at["timing_pnl"] == pytest.approx(((w - w.mean()) * unit).sum(), rel=1e-9, abs=1e-6)
    assert at["market_return"] == pytest.approx(np.prod(1 + m.iloc[1:]) - 1, rel=1e-12)
    # costs: trades decided in [first mark, last mark) + funding charged in (first mark, last mark]
    o = store.orders(acct, limit=100000)
    o = o[o["status"] == "filled"]
    bar = pd.to_datetime(o["reason"].str.removeprefix("bar "), utc=True)
    ok = (bar >= eq.index[0]) & (bar < eq.index[-1])
    fund = pd.read_sql("select ts, amount from funding where account=?", store.conn, params=(acct,))
    fts = pd.to_datetime(fund["ts"], utc=True, format="ISO8601")
    fees, spread = o.loc[ok, "fee"].sum(), o.loc[ok, "slippage_cost"].sum()
    funding = -fund.loc[(fts > eq.index[0]) & (fts <= eq.index[-1]), "amount"].sum()
    assert spread > 0 and len(fund) > 0
    assert at["fees"] == pytest.approx(fees) and at["spread"] == pytest.approx(spread)
    assert at["funding"] == pytest.approx(funding) and at["costs"] == pytest.approx(fees + spread + funding)
    assert at["passive_pnl"] + at["timing_pnl"] + at["selection_pnl"] - at["costs"] == pytest.approx(at["pnl"])
    # A window that starts on a funding stamp (price buffer trimmed): that stamp's funding is
    # already inside the starting mark, so it is not a cost of the window.
    s = pd.Timestamp(sorted(fts.unique())[-2])       # funding rows are per symbol: distinct stamps
    a2 = eng.status(close.loc[s:])[acct]["attribution"]
    assert a2["window_start"] == str(s)
    assert a2["pnl"] == pytest.approx(eq["equity"].iloc[-1] - eq["equity"].loc[s])
    ok2 = (bar >= s) & (bar < eq.index[-1])
    exp = (o.loc[ok2, "fee"].sum() + o.loc[ok2, "slippage_cost"].sum()
           - fund.loc[(fts > s) & (fts <= eq.index[-1]), "amount"].sum())
    assert a2["costs"] == pytest.approx(exp)
    # A halted symbol with an older, stale buffer must not stretch the window back.
    stale = pd.DataFrame({SYMS[0]: 50.0}, index=pd.date_range(end=close.index[0] - pd.Timedelta(minutes=5),
                                                             periods=300, freq="5min", tz="UTC"))
    c3 = pd.concat([stale, close]).reindex(columns=close.columns)
    c3.iloc[-48:, 0] = np.nan                         # symbol 0 stopped trading 4 hours ago
    a3 = eng.status(c3)[acct]["attribution"]
    assert pd.Timestamp(a3["window_start"]) >= close.index[0]
    assert a3["market_return"] == pytest.approx(
        np.prod(1 + basket_returns(close[SYMS[1:]]).reindex(eq.index).loc[pd.Timestamp(a3["window_start"]):].iloc[1:]) - 1)


def test_intraday_status_cli_prints_attribution_per_window(tmp_path, capsys, icfg):
    from tradinglab.cli import cmd_intraday_status

    def acc(start, mret, beta):
        return {"equity": 1e5, "pnl": 0.0, "return": 0.0, "trades": 1, "fees_paid": 0.0, "gross": 0.0,
                "attribution": {"window_start": start, "market_return": mret, "avg_exposure": 0.1, "beta": beta,
                                "passive_pnl": 1.0, "timing_pnl": 2.0, "selection_pnl": 3.0, "costs": 0.5}}

    icfg.paths.state_dir = str(tmp_path)
    (tmp_path / "intraday_status.json").write_text(json.dumps({
        "updated_at": "x", "last_bar": "y",
        "accounts": {"intraday:a": acc("2026-01-01 00:00:00+00:00", 0.01, 0.3),
                     "intraday:b": acc("2026-01-05 00:00:00+00:00", -0.02, None)}}))
    assert cmd_intraday_status(icfg, None) == 0
    out = capsys.readouterr().out
    assert "windows differ" in out and "2026-01-05 00:00:00+00:00" in out and "-2.00%" in out and "n/a" in out
