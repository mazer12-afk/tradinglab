"""Evaluate 5-minute policies on the walk-forward forecasts (development data only, net of costs)."""
import logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.intraday.data import load_intraday
from tradinglab.intraday.research import (forecast_score, hysteresis_positions, signal_to_weights,
                                          freeze_between_changes, backtest_weights)
from tradinglab.intraday.live import rule_outputs

logging.basicConfig(level=logging.WARNING)
IN = Path(sys.argv[1])
T0 = pd.Timestamp("2025-10-29", tz="UTC")
EVAL0 = pd.Timestamp("2023-08-01", tz="UTC")       # first month of forecasts warms the score normaliser up
MID = pd.Timestamp("2024-09-01", tz="UTC")
cfg = load_config()
ic = cfg.intraday
md = MarketData(cfg.paths.data_dir)
d = load_intraday(md, ic.symbols, "2022-07-01", ic.bar, end=T0, market="spot")
rows = np.arange(len(d.index))
preds = {f.stem[5:]: pd.read_parquet(f) for f in sorted(IN.glob("pred_*.parquet"))}
preds["tsmom48"] = pd.DataFrame(rule_outputs("tsmom48", d, rows), index=d.index, columns=d.symbols)
preds["tsmom288"] = pd.DataFrame(rule_outputs("tsmom288", d, rows), index=d.index, columns=d.symbols)


def selective(pred, H, enter, hold, exit_=0.0, long_only=False, norm=288 * 30):
    score = forecast_score(pred.reindex(d.index), H, norm)
    pos = hysteresis_positions(score, enter, exit_, hold)
    if long_only:
        pos = pos.clip(lower=0.0)
    w = signal_to_weights(pos.where(score.notna()), d, ic)
    return freeze_between_changes(pos, w)


def stats(w):
    bt = backtest_weights(w, d, ic, start=EVAL0, end=T0)
    r = bt.returns
    daily = (1 + r).groupby(r.index.floor("D")).prod() - 1
    out = {}
    for lab, a, b in [("A", EVAL0, MID), ("B", MID, T0), ("all", EVAL0, T0)]:
        x = daily[(daily.index >= a) & (daily.index < b)]
        eq = (1 + x).cumprod()
        out[f"{lab}_sharpe"] = x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan
        out[f"{lab}_ann"] = eq.iloc[-1] ** (365 / len(x)) - 1
    eq = (1 + daily).cumprod()
    out["mdd"] = (eq / eq.cummax() - 1).min()
    ww = w[(w.index >= EVAL0) & (w.index < T0)]
    out["turnover_yr"] = float(ww.diff().abs().sum(axis=1).sum() / ((T0 - EVAL0).days / 365))
    out["gross"] = float(ww.abs().sum(axis=1).mean())
    out["net"] = float(ww.sum(axis=1).mean())
    gross_ret = bt.gross_returns if hasattr(bt, "gross_returns") else None
    return out


variants = {
    # baselines (the live roster)
    "lgbm_1h": ("h12_1m", 12, 3.0, 12, False),
    "lgbm_4h": ("h48_1m", 48, 2.5, 24, False),
    "tsmom_4h": ("tsmom48", 1, 1.5, 12, False),
    # retraining interval for the 4h model
    "lgbm_4h_retrain_1w": ("h48_1w", 48, 2.5, 24, False),
    "lgbm_4h_retrain_3m": ("h48_3m", 48, 2.5, 24, False),
    # improvements / new
    "lgbm_4h_long": ("h48_1m", 48, 2.5, 24, True),
    "lgbm_4h_strict": ("h48_1m", 48, 3.0, 48, False),
    "lgbm_12h": ("h144_1m", 144, 2.5, 72, False),
    "lgbm_12h_long": ("h144_1m", 144, 2.5, 72, True),
    "lgbm_12h_strict": ("h144_1m", 144, 3.0, 144, False),
    "tsmom_4h_long": ("tsmom48", 1, 1.5, 12, True),
    "tsmom_1d": ("tsmom288", 1, 1.5, 48, False),
    "tsmom_1d_long": ("tsmom288", 1, 1.5, 48, True),
}
rows_out = {}
only = set(sys.argv[2:])
for name, (key, H, enter, hold, lo) in variants.items():
    if key not in preds or (only and name not in only):
        continue
    rows_out[name] = stats(selective(preds[key], H, enter, hold, long_only=lo))
    print(name, {k: round(v, 3) for k, v in rows_out[name].items()}, flush=True)
if not only:
    w = signal_to_weights(pd.DataFrame(1.0, index=d.index, columns=d.symbols), d, ic)
    rows_out["basket_long"] = stats(w)
t = pd.DataFrame(rows_out).T
pd.set_option("display.width", 250)
print(t.round(3).to_string())
t.to_csv(IN / ("eval_intervals.csv" if only else "eval.csv"))
