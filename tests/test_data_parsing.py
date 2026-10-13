"""Archive and exchange payload parsing (recorded formats, no network)."""

import pandas as pd
import pytest

from tradinglab.data.binance_vision import parse_funding_csv, parse_kline_csv
from tradinglab.data.exchanges import OKX, BinanceSpot, Coinbase, Kraken
from tradinglab.data.store import funding_schedule, hourly_to_bars


def test_kline_csv_ms_and_us_timestamps_and_header():
    ms = b"1567296000000,9588.74,9630.0,9579.34,9617.06,546.96,1567299599999,5257943.07,7428,261.13,2510479.64,0\n"
    us = b"1791763200000000,83773.12,83986.55,83721.98,83962.66,456.88,1791766799999999,38309932.52,72118,200.15,16781899.55,0\n"
    hdr = (b"open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
           b"taker_buy_quote_volume,ignore\n1791763200000,1,2,0.5,1.5,10,1791766799999,15,3,5,7,0\n")
    a, b, c = parse_kline_csv(ms), parse_kline_csv(us), parse_kline_csv(hdr)
    assert a.index[0] == pd.Timestamp("2019-09-01", tz="UTC")
    assert b.index[0] == pd.Timestamp("2026-10-12", tz="UTC")
    assert c.index[0] == pd.Timestamp("2026-10-12", tz="UTC") and c["close"].iloc[0] == 1.5
    assert b["close"].iloc[0] == pytest.approx(83962.66)


def test_funding_csv_and_schedule():
    raw = b"calc_time,funding_interval_hours,last_funding_rate\n1788220800005,8,0.0001\n1788249600000,8,0.0002\n"
    f = parse_funding_csv(raw)
    assert list(f.index) == [pd.Timestamp("2026-09-01 00:00", tz="UTC"), pd.Timestamp("2026-09-01 08:00", tz="UTC")]
    sched, est = funding_schedule(f["funding_rate"], pd.Timestamp("2026-08-31 16:00", tz="UTC"),
                                  pd.Timestamp("2026-09-02", tz="UTC"), 0.00005)
    assert sched.loc[pd.Timestamp("2026-08-31 16:00", tz="UTC")] == 0.00005   # before first print → baseline
    assert sched.loc[pd.Timestamp("2026-09-01 08:00", tz="UTC")] == 0.0002
    assert est.loc[pd.Timestamp("2026-09-01 16:00", tz="UTC")]                  # after last print → estimate
    assert sched.loc[pd.Timestamp("2026-09-01 16:00", tz="UTC")] == pytest.approx(0.00015)


def test_hourly_to_daily_drops_incomplete_bars(hourly):
    h = hourly["AAAUSDT"].iloc[:24 * 3 + 5]
    d = hourly_to_bars(h, "1d")
    assert len(d) == 3
    assert d.index[0] == pd.Timestamp("2024-01-02", tz="UTC")       # labelled by close time
    assert d["close"].iloc[0] == h["close"].iloc[23]


def test_exchange_kline_parsers():
    now_old = 1_600_000_000_000
    b = BinanceSpot.parse_klines([[now_old, "1", "2", "0.5", "1.5", "10", now_old + 3599999, "15", 3, "5", "7", "0"]])
    assert b["close"].iloc[0] == 1.5 and b.index[0] == pd.Timestamp(now_old, unit="ms", tz="UTC")
    cb = Coinbase.parse_klines([[1_600_000_000, 0.5, 2, 1, 1.5, 10]])
    assert cb["open"].iloc[0] == 1 and cb["quote_volume"].iloc[0] == 15
    kr = Kraken.parse_klines({"error": [], "result": {"XXBTZUSD": [[1_600_000_000, "1", "2", "0.5", "1.5", "1.2", "10", 3]],
                                                       "last": 1}})
    assert kr["close"].iloc[0] == 1.5 and kr["quote_volume"].iloc[0] == pytest.approx(12)
    ok = OKX.parse_klines({"data": [[str(now_old), "1", "2", "0.5", "1.5", "10", "15", "15", "1"],
                                    [str(now_old + 3600000), "1", "2", "0.5", "1.5", "10", "15", "15", "0"]]})
    assert len(ok) == 1  # unconfirmed (still forming) candle dropped
    assert Kraken().pair("BTCUSDT") == "XBTUSD" and Coinbase.product("ETHUSDT") == "ETH-USD"
