"""Live-endpoint integration tests. Skipped unless RUN_NETWORK_TESTS=1.

Run them on the machine that will trade to verify connectivity:

    RUN_NETWORK_TESTS=1 pytest -q tests/test_network.py
"""

import os

import pandas as pd
import pytest

from tradinglab.data.binance_vision import BinanceVision
from tradinglab.data.exchanges import default_exchanges

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network tests disabled")


def test_archive_lists_recent_daily_files(tmp_path):
    bv = BinanceVision(tmp_path)
    start = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=5)
    df = bv.klines("BTCUSDT", "1h", start)
    assert not df.empty and df.index[-1] >= start


@pytest.mark.parametrize("exchange", default_exchanges(), ids=lambda e: e.name)
def test_exchange_quote_and_klines(exchange):
    try:
        q = exchange.quote("BTCUSDT")
    except Exception as exc:  # noqa: BLE001 - geo-blocked venues are allowed to fail
        pytest.skip(f"{exchange.name} unreachable here: {exc}")
    assert 0 < q.bid <= q.ask < q.bid * 1.01
    k = exchange.klines_1h("BTCUSDT", pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=30))
    assert len(k) >= 20
    assert k.index[-1] + pd.Timedelta(hours=1) <= pd.Timestamp.now(tz="UTC")   # closed bars only
    assert (k["high"] >= k["low"]).all()
