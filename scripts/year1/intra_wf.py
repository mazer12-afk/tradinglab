"""5-minute book development: walk-forward LightGBM forecasts on spot data before T0 (the live recipe)."""
import dataclasses, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.intraday.data import load_intraday
from tradinglab.intraday.features import build_features
from tradinglab.intraday.models import LGBMForecaster
from tradinglab.intraday.research import train_rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("wf")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
T0 = pd.Timestamp("2025-10-29", tz="UTC")
FIRST = pd.Timestamp("2023-07-01", tz="UTC")
cfg = load_config()
ic = cfg.intraday
md = MarketData(cfg.paths.data_dir)
d = load_intraday(md, ic.symbols, "2022-07-01", ic.bar, end=T0, market="spot")
log.info("data %s → %s, %s", d.index[0], d.index[-1], d.close.shape)
d.close.to_parquet(OUT / "close.parquet")


def steps(freq):
    if freq == "1w":
        return list(pd.date_range(FIRST, T0, freq="7D"))
    m = int(freq[:-1])
    out, t = [], FIRST
    while t < T0:
        out.append(t); t = t + pd.DateOffset(months=m)
    return out


def train(fs, H, end):
    """The simulator's / live recipe (intraday.train.train_live_intraday): 150 rounds, lr 0.02, train + valid rows."""
    icH = dataclasses.replace(ic, horizon_bars=H)
    tr, va = train_rows(fs, icH, end)
    X, y, _, a = fs.rows(np.concatenate([tr, va]))
    model = LGBMForecaster(learning_rate=0.02)
    model.params["learning_rate"] = 0.02
    Xa = np.column_stack([X, a])
    model.booster = lgb.train(model.params, lgb.Dataset(Xa, y, categorical_feature=[Xa.shape[1] - 1]),
                              num_boost_round=150)
    model.best_iteration = 150
    return model


runs = [(12, "1m"), (48, "1m"), (144, "1m"), (48, "3m"), (48, "1w")]
feats = {}
for H, freq in runs:
    if H not in feats:
        feats.clear()
        feats[H] = build_features(d, H)
    fs = feats[H]
    st = steps(freq)
    preds = []
    t0 = time.time()
    for i, s in enumerate(st):
        e = st[i + 1] if i + 1 < len(st) else T0
        m = train(fs, H, s)
        lo, hi = np.searchsorted(fs.index, s), np.searchsorted(fs.index, e)
        X, _, tt, aa = fs.rows(np.arange(lo, hi), need_y=False)
        out = np.full((hi - lo, len(fs.symbols)), np.nan)
        out[tt - lo, aa] = m.predict(X, asset=aa)
        preds.append(pd.DataFrame(out, index=fs.index[lo:hi], columns=fs.symbols))
    p = pd.concat(preds)
    p.to_parquet(OUT / f"pred_h{H}_{freq}.parquet")
    log.info("h%d %s: %d steps in %.0fs", H, freq, len(st), time.time() - t0)
