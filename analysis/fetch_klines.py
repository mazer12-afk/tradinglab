"""Fetch closed spot klines (data-api.binance.vision) for the experiment window into parquet."""
import sys, time
from pathlib import Path
import pandas as pd
import requests

OUT = Path(sys.argv[1])
START = pd.Timestamp("2026-10-14 00:00", tz="UTC")
END = pd.Timestamp("2026-10-30 00:00", tz="UTC")
SETS = {
    "5m30": ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'TRXUSDT', 'LINKUSDT',
           'AVAXUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'ATOMUSDT', 'NEARUSDT', 'UNIUSDT', 'ETCUSDT',
           'FILUSDT', 'AAVEUSDT', 'HBARUSDT', 'ALGOUSDT', 'VETUSDT', 'INJUSDT', 'SUIUSDT', 'APTUSDT', 'ARBUSDT',
           'OPUSDT', 'ICPUSDT', 'LDOUSDT'],
    "5m": ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT',
           'LTCUSDT', 'DOTUSDT', 'BCHUSDT'],
    "1h": ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'TRXUSDT', 'LINKUSDT',
           'AVAXUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'ATOMUSDT', 'NEARUSDT', 'UNIUSDT', 'ETCUSDT',
           'FILUSDT', 'AAVEUSDT', 'HBARUSDT', 'ALGOUSDT', 'VETUSDT', 'INJUSDT', 'SUIUSDT', 'APTUSDT', 'ARBUSDT',
           'OPUSDT', 'ICPUSDT', 'LDOUSDT'],
}
s = requests.Session()
for interval, syms in [(k, v) for k, v in SETS.items() if k in (sys.argv[2:] or SETS)]:
    closes = {}
    for sym in syms:
        rows, t = [], int(START.timestamp() * 1000)
        while t < END.timestamp() * 1000:
            for attempt in range(5):
                try:
                    r = s.get("https://data-api.binance.vision/api/v3/klines",
                              params={"symbol": sym, "interval": interval[:2], "startTime": t, "limit": 1000}, timeout=20)
                    r.raise_for_status()
                    break
                except Exception as e:
                    time.sleep(2 ** attempt)
            else:
                raise SystemExit(f"failed {sym}")
            data = r.json()
            if not data:
                break
            rows += data
            t = data[-1][0] + 1
        df = pd.DataFrame(rows, columns=list(range(12)))
        idx = pd.to_datetime(df[0], unit="ms", utc=True)
        closes[sym] = pd.Series(df[4].astype(float).values, index=idx)
        print(interval, sym, len(df), idx.min(), idx.max(), flush=True)
    pd.DataFrame(closes).sort_index().to_parquet(OUT / f"close_{interval}.parquet")
