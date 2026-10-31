"""Edge per trade of the 5-minute strategies before costs vs what a trade costs (development data, Aug 2023 → Oct 2025)."""
import dataclasses, logging, sys
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
T0 = pd.Timestamp("2025-10-29", tz="UTC"); EV = pd.Timestamp("2023-08-01", tz="UTC")
cfg = load_config(); ic = cfg.intraday
md = MarketData(cfg.paths.data_dir)
d = load_intraday(md, ic.symbols, "2022-07-01", ic.bar, end=T0, market="spot")
free = dataclasses.replace(ic, fee_bps=0.0, slippage_bps={}, default_slippage_bps=0.0)
nofund = dataclasses.replace(d, funding=d.funding * 0.0)
taker_side = float(np.mean(ic.cost_bps(d.symbols)))
preds = {k: pd.read_parquet(IN / f"pred_{k}.parquet") for k in ("h12_1m", "h48_1m", "h144_1m")}
preds["tsmom48"] = pd.DataFrame(rule_outputs("tsmom48", d, np.arange(len(d.index))), index=d.index, columns=d.symbols)


def sel(pred, H, enter, hold):
    score = forecast_score(pred.reindex(d.index), H, 288 * 30)
    pos = hysteresis_positions(score, enter, 0.0, hold)
    return freeze_between_changes(pos, signal_to_weights(pos.where(score.notna()), d, ic))


rows = {}
for name, key, H, enter, hold in [("lgbm_1h", "h12_1m", 12, 3.0, 12), ("lgbm_4h", "h48_1m", 48, 2.5, 24),
                                  ("lgbm_12h", "h144_1m", 144, 2.5, 72), ("tsmom_4h", "tsmom48", 1, 1.5, 12)]:
    w = sel(preds[key], H, enter, hold)
    yrs = (T0 - EV).days / 365
    net = backtest_weights(w, d, ic, start=EV, end=T0).returns
    gross = backtest_weights(w, nofund, free, start=EV, end=T0).returns
    ww = w[(w.index >= EV) & (w.index < T0)]
    traded = float(ww.diff().abs().sum(axis=1).sum())          # notional traded / capital, both sides counted
    g = float(gross.sum()); n = float(net.sum())
    rows[name] = {"gross_per_yr": g / yrs, "net_per_yr": n / yrs, "traded_per_yr": traded / yrs,
                  "edge_bp_per_side": 1e4 * g / traded, "taker_cost_bp_per_side": taker_side,
                  "hold_hours_avg": None}
t = pd.DataFrame(rows).T
pd.set_option("display.width", 200)
print(t.drop(columns="hold_hours_avg").round(3).to_string())
