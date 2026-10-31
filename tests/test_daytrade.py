"""Day-trading simulator: fill rules on hand-built minute bars, costs, portfolio book."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglab.daytrade.sim import Costs, Minutes, book_stats, funding_cost, run_book, simulate_trades

T0 = pd.Timestamp("2025-01-01", tz="UTC")
COSTS = Costs(maker_bps=2.0, taker_bps=5.0, impact_bps=1.0, through_bps=1.0, half_spread={"AAA": 2e-4})


def bars(rows):
    """rows: list of (open, high, low, close) for coin AAA, one per minute."""
    idx = pd.date_range(T0, periods=len(rows), freq="1min")
    a = np.array(rows, dtype=float)
    f = [pd.DataFrame({"AAA": a[:, i]}, index=idx) for i in range(4)]
    return Minutes(*f)


def intent(m, s, side=1, entry="market", price=np.nan, ttl=0, tp=0.0, sl=0.0, hold=3, size=0.5):
    return pd.DataFrame([{"t": m.index[s], "symbol": "AAA", "side": side, "size": size, "entry": entry,
                          "price": price, "ttl": ttl, "tp": tp, "sl": sl, "max_hold": hold}])


FLAT = [(100, 100.1, 99.9, 100)] * 10


def test_market_entry_and_time_exit_pay_spread_impact_and_taker_fees():
    rows = list(FLAT)
    rows[2] = (101, 101.2, 100.8, 101)          # decision at minute 1 → fill at minute 2's open
    rows[5] = (103, 103.1, 102.9, 103)          # exit at the open of minute 2 + 3
    m = bars(rows)
    r = simulate_trades(m, intent(m, 1, hold=3), COSTS).iloc[0]
    slip = 2e-4 + 1e-4
    assert r.filled and r.entry_time == m.index[2] and r.exit_time == m.index[5] and r.exit_kind == "time"
    assert r.entry_px == pytest.approx(101 * (1 + slip)) and r.exit_px == pytest.approx(103 * (1 - slip))
    assert r.fees == pytest.approx(10e-4)
    short = simulate_trades(m, intent(m, 1, side=-1, hold=3), COSTS).iloc[0]
    assert short.gross == pytest.approx(-(103 * (1 + slip) / (101 * (1 - slip)) - 1), rel=1e-6)


def test_limit_fills_only_when_traded_through_and_expires():
    rows = list(FLAT)
    rows[2] = (100, 100.1, 99.0, 99.5)           # touches the limit exactly: not filled
    m = bars(rows)
    r = simulate_trades(m, intent(m, 0, entry="limit", price=99.0, ttl=5), COSTS).iloc[0]
    assert not r.filled                           # expired after 5 minutes without trading through
    rows[4] = (99.5, 99.6, 98.9, 99.2)           # trades through 99.0 by > 1 bp
    m = bars(rows)
    r = simulate_trades(m, intent(m, 0, entry="limit", price=99.0, ttl=5, hold=3), COSTS).iloc[0]
    assert r.filled and r.entry_time == m.index[4] and r.entry_px == 99.0
    assert r.fees == pytest.approx(2e-4 + 5e-4)  # maker in, taker (time exit) out
    rows[4] = (98.5, 98.6, 98.4, 98.5)           # opened below the resting limit: filled at the open
    m = bars(rows)
    assert simulate_trades(m, intent(m, 0, entry="limit", price=99.0, ttl=5), COSTS).iloc[0].entry_px == 98.5
    late = simulate_trades(m, intent(m, 0, entry="limit", price=99.0, ttl=3), COSTS).iloc[0]
    assert not late.filled                        # minute 4 is after the 3-minute life of the order


def test_stop_before_take_profit_in_the_same_minute_and_gaps():
    rows = list(FLAT)
    rows[3] = (100, 102, 98, 100)                 # both 101 TP and 99 SL inside one minute → stop first
    m = bars(rows)
    r = simulate_trades(m, intent(m, 0, tp=0.01, sl=0.01, hold=8), COSTS).iloc[0]
    e = 100 * (1 + 3e-4)
    assert r.exit_kind == "stop" and r.exit_px == pytest.approx(e * 0.99 * (1 - 3e-4))
    rows[3] = (97, 97.5, 96.5, 97)                # opens below the stop: filled at the open
    m = bars(rows)
    r = simulate_trades(m, intent(m, 0, tp=0.01, sl=0.01, hold=8), COSTS).iloc[0]
    assert r.exit_kind == "stop" and r.exit_px == pytest.approx(97 * (1 - 3e-4))
    rows[3] = (100, 101.5, 100, 101)              # take-profit traded through: maker exit at the TP
    m = bars(rows)
    r = simulate_trades(m, intent(m, 0, tp=0.01, sl=0.01, hold=8), COSTS).iloc[0]
    assert r.exit_kind == "take-profit" and r.exit_px == pytest.approx(e * 1.01) and r.fees == pytest.approx(7e-4)


def test_limit_entry_exits_are_checked_from_the_next_minute():
    rows = list(FLAT)
    rows[2] = (100, 101.5, 98.8, 101)             # fills the 99 limit and also spans the TP: order unknown
    m = bars(rows)
    # TP at 99 × 1.02 = 100.98: inside the fill minute's range, above every later minute's high (100.1)
    r = simulate_trades(m, intent(m, 0, entry="limit", price=99.0, ttl=5, tp=0.02, hold=4), COSTS).iloc[0]
    assert r.entry_time == m.index[2] and r.exit_kind == "time"


def test_fills_never_use_the_decision_minute_or_earlier():
    rows = list(FLAT)
    m = bars(rows)
    base = simulate_trades(m, intent(m, 4, hold=3), COSTS).iloc[0]
    rows[0] = rows[1] = rows[4] = (50, 200, 10, 60)  # wild prices up to and including the decision minute
    m2 = bars(rows)
    r = simulate_trades(m2, intent(m2, 4, hold=3), COSTS).iloc[0]
    assert r.entry_px == base.entry_px and r.exit_px == base.exit_px and r.entry_time > m2.index[4]


def test_book_one_trade_per_coin_gross_cap_and_compounding():
    idx = pd.date_range(T0, periods=4 * 1440, freq="1min")
    t = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "CCC", "AAA"],
        "side": 1, "size": [0.5, 0.5, 1.0, 1.0, 0.5], "entry": "market",
        "filled": True,
        "entry_time": [idx[10], idx[20], idx[15], idx[16], idx[1500]],
        "exit_time": [idx[100], idx[110], idx[50], idx[60], idx[1600]],
        "gross": [0.02, 0.02, 0.01, 0.01, -0.01], "fees": [0.001] * 5,
    })
    b = run_book(t, capital=100_000.0, max_gross=2.0)
    # 2nd AAA overlaps the open AAA trade; CCC would take gross to 2.5x
    assert list(b.trades["entry_time"]) == [idx[10], idx[15], idx[1500]]
    first = 0.5 * 100_000 * 0.019 + 1.0 * 100_000 * 0.009
    assert b.trades["pnl"].iloc[:2].sum() == pytest.approx(first)
    assert b.trades["notional"].iloc[2] == pytest.approx(0.5 * (100_000 + first))
    assert b.daily.iloc[0] == pytest.approx(first / 100_000)
    s = book_stats(b)
    assert s["trades"] == 3 and s["win_rate"] == pytest.approx(2 / 3)


def test_funding_is_charged_per_settlement_held_through():
    idx = pd.date_range("2025-01-01 07:00", periods=4, freq="2h", tz="UTC")
    t = pd.DataFrame({"symbol": ["AAA", "AAA"], "side": [1, -1], "entry_time": [idx[0], idx[0]],
                      "exit_time": [idx[1], idx[1]]})
    rates = {"AAA": pd.Series([0.0003, 0.0005], index=pd.DatetimeIndex(["2025-01-01 08:00", "2025-01-01 16:00"], tz="UTC"))}
    np.testing.assert_allclose(funding_cost(t, rates), [0.0003, -0.0003])
