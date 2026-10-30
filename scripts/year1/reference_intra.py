"""Test-year reference for the pruned 5-minute book: monthly-retrained LightGBM (live recipe) + rules, net of costs."""
import dataclasses, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.intraday.data import load_intraday
from tradinglab.intraday.features import build_features
from tradinglab.intraday.models import LGBMForecaster
from tradinglab.intraday.research import (train_rows, forecast_score, hysteresis_positions, signal_to_weights,
                                          freeze_between_changes, backtest_weights)
from tradinglab.intraday.live import rule_outputs

logging.basicConfig(level=logging.WARNING)
T0, T1 = pd.Timestamp("2025-10-29", tz="UTC"), pd.Timestamp("2026-10-29", tz="UTC")
cfg = load_config(); ic = cfg.intraday
md = MarketData(cfg.paths.data_dir)
d = load_intraday(md, ic.symbols, "2024-08-01", ic.bar, end=T1, market="spot")
steps = [T0] + list(pd.date_range(pd.Timestamp("2025-11-01", tz="UTC"), T1, freq="MS", inclusive="left"))
preds = {}
for H in (12, 48):
    fs = build_features(d, H)
    parts = []
    for i, s in enumerate(steps):
        e = steps[i + 1] if i + 1 < len(steps) else T1
        tr, va = train_rows(fs, dataclasses.replace(ic, horizon_bars=H), s)
        X, y, _, a = fs.rows(np.concatenate([tr, va]))
        m = LGBMForecaster(learning_rate=0.02); m.params["learning_rate"] = 0.02
        Xa = np.column_stack([X, a])
        m.booster = lgb.train(m.params, lgb.Dataset(Xa, y, categorical_feature=[Xa.shape[1] - 1]), num_boost_round=150)
        m.best_iteration = 150
        lo, hi = np.searchsorted(fs.index, s), np.searchsorted(fs.index, e)
        Xp, _, tt, aa = fs.rows(np.arange(lo, hi), need_y=False)
        out = np.full((hi - lo, len(fs.symbols)), np.nan)
        out[tt - lo, aa] = m.predict(Xp, asset=aa)
        parts.append(pd.DataFrame(out, index=fs.index[lo:hi], columns=fs.symbols))
    # earlier forecasts for the score normaliser: the first model's in-window history is not available live
    # either (the live engine warms up on the buffer with the current model) → warm up with the T0 model
    preds[H] = pd.concat(parts)
    del fs
rows = np.arange(len(d.index))
tsm = pd.DataFrame(rule_outputs("tsmom48", d, rows), index=d.index, columns=d.symbols)


def sel(pred, H, enter, hold):
    score = forecast_score(pred.reindex(d.index), H, 288 * 30)
    pos = hysteresis_positions(score, enter, 0.0, hold)
    return freeze_between_changes(pos, signal_to_weights(pos.where(score.notna()), d, ic))


out = {}
EV = T0 + pd.Timedelta(days=2)
for name, w in [("lgbm_1h", sel(preds[12], 12, 3.0, 12)), ("lgbm_4h", sel(preds[48], 48, 2.5, 24)),
                ("tsmom_4h", sel(tsm, 1, 1.5, 12)),
                ("basket_long_5m", signal_to_weights(pd.DataFrame(1.0, index=d.index, columns=d.symbols), d, ic))]:
    bt = backtest_weights(w, d, ic, start=EV, end=T1)
    r = bt.returns
    daily = (1 + r).groupby(r.index.floor("D")).prod() - 1
    eq = (1 + daily).cumprod()
    free = dataclasses.replace(ic, fee_bps=0.0, slippage_bps={}, default_slippage_bps=0.0)
    g = backtest_weights(w, d, free, start=EV, end=T1).returns
    gd = (1 + g).groupby(g.index.floor("D")).prod() - 1
    ww = w[(w.index >= EV) & (w.index < T1)]
    out[name] = {"return": eq.iloc[-1] - 1, "sharpe": daily.mean() / daily.std() * np.sqrt(365),
                 "mdd": (eq / eq.cummax() - 1).min(), "return_before_costs": (1 + gd).prod() - 1,
                 "turnover": float(ww.diff().abs().sum(axis=1).sum()), "net_exposure": float(ww.sum(axis=1).mean())}
    eq.rename(name).to_csv(Path(sys.argv[1]).with_name(f"curve_{name}.csv"))
t = pd.DataFrame(out).T
print(t.round(3).to_string())
t.to_csv(sys.argv[1])
