"""Round 3: LightGBM meta-model on extreme-move events with futures-metrics features, retrained monthly (walk-forward)."""
import json, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from load import load_minutes
from signals import event_frame, intents_from_mask
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.daytrade.sim import Costs, simulate_trades, funding_cost
from tradinglab.daytrade.research import Registry, evaluate

FOLDS3 = [("F1", "2023-02-01", "2023-08-01"), ("F2", "2023-08-01", "2024-09-01"), ("F3", "2024-09-01", "2025-10-29")]
REG = Registry(sys.argv[4])
MET = Path(sys.argv[3])
m = load_minutes("2022-07-01", "2025-10-30")
ev = event_frame(m, MET)
grid, syms = ev["grid"], m.symbols
lc = np.log(m.close.ffill())
L = lc.reindex(grid)
H = 240
fwd = (lc.shift(-H).reindex(grid) - L)                                # label: 4h forward log return from the decision
met = {s: pd.read_parquet(MET / f"{s}.parquet").reindex(grid) for s in syms}
md = MarketData(load_config().paths.data_dir)
fr = pd.DataFrame({s: md.funding_8h(s, pd.Timestamp("2022-06-01")) for s in syms})
fr.index = fr.index.round("min")
fund = fr.reindex(grid, method="ffill")
mk = lc.diff().mean(axis=1).fillna(0.0).cumsum().reindex(grid)
feat = {
    "z": ev["z"], "iz": ev["iz"], "abs_z": ev["z"].abs(), "doi_z": ev["doi_z"],
    "sig5": np.log(ev["sig"]), "sig5_rel": np.log(ev["sig"] / ev["sig"].rolling(288 * 7, min_periods=288).mean()),
    "ret_1h": L - L.shift(12), "ret_4h": L - L.shift(48), "ret_1d": L - L.shift(288),
    "doi_1h": pd.DataFrame({s: np.log(met[s]["sum_open_interest_value"]) for s in syms}).diff(12),
    "doi_1d": pd.DataFrame({s: np.log(met[s]["sum_open_interest_value"]) for s in syms}).diff(288),
    "taker_5m": pd.DataFrame({s: np.log(met[s]["sum_taker_long_short_vol_ratio"]) for s in syms}),
    "taker_1h": pd.DataFrame({s: np.log(met[s]["sum_taker_long_short_vol_ratio"]) for s in syms}).rolling(12, min_periods=8).mean(),
    "top_ls": pd.DataFrame({s: np.log(met[s]["sum_toptrader_long_short_ratio"]) for s in syms}),
    "top_ls_1h": pd.DataFrame({s: np.log(met[s]["sum_toptrader_long_short_ratio"]) for s in syms}).diff(12),
    "crowd_ls": pd.DataFrame({s: np.log(met[s]["count_long_short_ratio"]) for s in syms}),
    "funding": fund,
}
bc = lambda x: pd.DataFrame(np.repeat(np.asarray(x)[:, None], len(syms), axis=1), index=grid, columns=syms)
feat.update({"mkt_1h": bc(mk - mk.shift(12)), "mkt_1d": bc(mk - mk.shift(288)), "mkt_3d": bc(mk - mk.shift(864)),
             "hour": bc(grid.hour), "dow": bc(grid.dayofweek),
             "to_settle": bc(((8 - grid.hour % 8) * 60 - grid.minute) % 480)})
names = list(feat)
event = ev["z"].abs() > 3.0
st = event.stack(); st = st[st]
idx = st.index
X = np.column_stack([feat[n].stack().reindex(idx).to_numpy(dtype="float64") for n in names])
y = fwd.stack().reindex(idx).to_numpy()
sym_code = pd.Categorical(idx.get_level_values(1), categories=syms).codes
X = np.column_stack([X, sym_code])
t_ev = idx.get_level_values(0)
print("events", len(idx), "features", len(names), flush=True)
params = {"objective": "huber", "alpha": 0.02, "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
          "num_threads": 4, "seed": 0, "deterministic": True}
pred = np.full(len(idx), np.nan)
months = pd.date_range("2023-02-01", "2025-11-01", freq="MS", tz="UTC")
t0 = time.time()
for a, b in zip(months[:-1], months[1:]):
    tr = (t_ev < a - pd.Timedelta(minutes=H + 10)) & np.isfinite(y)          # labels complete before the month
    te = (t_ev >= a) & (t_ev < b)
    if te.sum() == 0:
        continue
    model = lgb.train(params, lgb.Dataset(X[tr], y[tr], categorical_feature=[X.shape[1] - 1]), num_boost_round=200)
    pred[te] = model.predict(X[te])
print(f"walk-forward done in {time.time()-t0:.0f}s; corr(pred, y) by fold:", flush=True)
for f, a, b in FOLDS3:
    s = (t_ev >= pd.Timestamp(a, tz="UTC")) & (t_ev < pd.Timestamp(b, tz="UTC")) & np.isfinite(pred) & np.isfinite(y)
    print(f, int(s.sum()), round(float(np.corrcoef(pred[s], y[s])[0, 1]), 3), flush=True)
P = pd.Series(pred, index=idx).unstack().reindex(index=grid, columns=syms)
costs = Costs(half_spread=json.load(open("/home/user/tradinglab/config/sim_spreads.json"))["half_spread"])
rates = {s: md.funding_8h(s, pd.Timestamp("2022-08-01")) for s in syms}
for theta in (0.0015, 0.0030):
    for entry, off, ttl in (("market", 0.0, 0), ("limit", 0.5, 15)):
        longs, shorts = P > theta, P < -theta
        it = pd.concat([intents_from_mask(ev, longs, 1, entry, off, ttl, H), intents_from_mask(ev, shorts, -1, entry, off, ttl, H)],
                       ignore_index=True)
        it["t"] = pd.DatetimeIndex(it["t"])
        trd = simulate_trades(m, it, costs)
        trd["funding"] = funding_cost(trd, rates)
        folds, pooled = evaluate(trd, rates, max_gross=1.0, folds=FOLDS3)
        p = {"model": "lgbm events |z|>3, 23 features, monthly walk-forward", "theta": theta, "entry": entry, "offset": off, "ttl": ttl, "hold": H}
        row = REG.add("ml_event_filter", p, folds, pooled)
        print(f"#{row['trial_no']:>3} theta {theta*1e4:.0f}bp {entry} folds " +
              " ".join(f"{folds[f]['ann_return']:+.0%}" if folds[f].get('trades') else "n/a" for f, _, _ in FOLDS3) +
              f" sharpe {row['sharpe']:.2f} dsr {row['dsr']:.2f} net {row['net_bp']:.1f}bp trades {row['trades']} "
              f"(long {int((trd.side > 0).sum())}/short {int((trd.side < 0).sum())}) accepted {row['accepted']}", flush=True)
print("registry trials so far:", REG.n_trials())
