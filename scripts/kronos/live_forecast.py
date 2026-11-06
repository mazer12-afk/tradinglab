"""Kronos-small's forecast for the next 24 hourly candles of a coin from live Binance data (www.binance.com).

    python scripts/kronos/live_forecast.py [--symbol BTCUSDT] [--paths 100] [--out reports/kronos/live_BTCUSDT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab import kronos as K
from tradinglab.daytrade.live import BinanceLive

CONTEXT, HORIZON = 488, 24


def hourly_live(api: BinanceLive, symbol: str, n: int) -> pd.DataFrame:
    """The last ``n`` closed hourly candles (OHLCV and quote volume) of a USD-M perpetual."""
    now = api.server_time()
    end = now.floor("h")
    rows, t = [], int((end - pd.Timedelta(hours=n)).value // 10**6)
    while t < int(end.value // 10**6):
        r = api._get("/fapi/v1/klines", {"symbol": symbol, "interval": "1h", "startTime": t,
                                         "endTime": int(end.value // 10**6) - 1, "limit": 1500})
        if not r:
            break
        rows += r
        t = int(r[-1][0]) + 3_600_000
    df = pd.DataFrame([[x[0], *x[1:6], x[7]] for x in rows], columns=["t", *K.COLUMNS])
    df.index = pd.to_datetime(df.pop("t").astype("int64"), unit="ms", utc=True)
    df = df.astype("float64")
    return df[~df.index.duplicated()].iloc[-n:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    api = BinanceLive()
    c = hourly_live(api, a.symbol, CONTEXT)
    assert len(c) == CONTEXT and (c.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all(), "gap in candles"
    import torch

    tok, mdl = K.load(threads=4)
    z = K.Sampler(tok, mdl).forecast([c], HORIZON, a.paths, generator=torch.Generator().manual_seed(0))[0]
    closes = z[:, :, 3]
    future = pd.date_range(c.index[-1], periods=HORIZON + 1, freq="1h")[1:]
    q = np.percentile(closes, [10, 50, 90], axis=0)
    out = {"symbol": a.symbol, "model": K.MODEL, "made_at": str(pd.Timestamp.now(tz="UTC")), "last_candle": str(c.index[-1]),
           "last_close": float(c["close"].iloc[-1]), "paths": int(a.paths),
           "history": {"t": [str(t) for t in c.index[-72:]], **{k: c[k].iloc[-72:].round(6).tolist() for k in ("open", "high", "low", "close")}},
           "future_t": [str(t) for t in future], "path_close": closes.round(6).tolist(),
           "p10": q[0].round(6).tolist(), "p50": q[1].round(6).tolist(), "p90": q[2].round(6).tolist(),
           "share_of_paths_up_24h": float((closes[:, -1] > c["close"].iloc[-1]).mean())}
    path = Path(a.out or f"reports/kronos/live_{a.symbol}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("history", "path_close", "future_t", "p10", "p50", "p90")}, indent=1))
    print("24 h ahead: p10 %.2f  p50 %.2f  p90 %.2f" % (q[0][-1], q[1][-1], q[2][-1]))


if __name__ == "__main__":
    main()
