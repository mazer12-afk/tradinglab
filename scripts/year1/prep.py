"""Data for the year-1 report page."""
import json, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData

S = Path("/tmp/claude-0/-home-user-tradinglab/b077bfc9-cfc5-5b41-a1e7-c2b38a175fb7/scratchpad")
R = Path("/home/user/tradinglab/reports/simulations/year1")
T0, T1 = pd.Timestamp("2025-10-29", tz="UTC"), pd.Timestamp("2026-10-29", tz="UTC")
cfg = load_config()
md = MarketData(cfg.paths.data_dir)

# market: daily opens → index
px = pd.DataFrame({s: md.hourly(s, "2025-10-01")["open"] for s in cfg.universe.symbols})
px = px[(px.index >= T0) & (px.index <= T1)].resample("D").first()
ew = (px.pct_change().mean(axis=1).fillna(0) + 1).cumprod() * 100
btc = px["BTCUSDT"] / px["BTCUSDT"].iloc[0] * 100

# simulated accounts: wealth (equity + vault) at daily marks
c = sqlite3.connect(f"file:{S}/sim/year1/lab.sqlite?immutable=1", uri=True)
e = pd.read_sql("select account, ts, equity, net from equity", c)
e["ts"] = pd.to_datetime(e.ts, utc=True, format="ISO8601")
v = pd.read_sql("select account, ts, amount, mark_ts, equity_marked from vault", c)
v["mark_ts"] = pd.to_datetime(v.mark_ts, utc=True, format="ISO8601")
curves = {}
for a, g in e.groupby("account"):
    w = g.set_index("ts")["equity"].sort_index()
    for _, r in v[v.account == a].iterrows():
        w[w.index > r.mark_ts] += r.amount
    curves[a] = (w.resample("D").last().reindex(ew.index, method="ffill").bfill() / 1000).round(3)

acc = pd.read_csv(R / "accounts.csv", dtype={"vault": str})
summ = pd.read_csv(R / "summary.csv").set_index("account")
acc = acc.join(summ[["fills", "fees", "spread", "funding"]], on="account")
ref = pd.read_csv(R / "test_year_backtest_daily_all.csv", index_col=0)
dev = pd.read_csv(R / "dev_daily_candidates.csv", index_col=0)
oos = pd.read_csv("/home/user/tradinglab/reports/research/oos_returns.csv", index_col=0, parse_dates=True)
for n in ("dl_lstm", "dl_mom_trans", "dl_deepm"):
    for lab, a, b in [("2021-23", "2021-01-01", "2023-10-29"), ("2023-25", "2023-10-29", "2025-10-29"), ("2021-25", "2021-01-01", "2025-10-29")]:
        x = oos[n][(oos.index >= pd.Timestamp(a, tz="UTC")) & (oos.index < pd.Timestamp(b, tz="UTC"))].dropna()
        dev.loc[n, f"sharpe_{lab}"] = x.mean() / x.std() * np.sqrt(365)

ctrl = acc[acc.vault == "0"].copy()
ctrl["name"] = ctrl.account.str.replace("shadow:", "", regex=False)
sim_ret = dict(zip(ctrl.name, ctrl.ret))

ROLE = {"tsmom": "kept", "tsmom_lf": "kept", "macd_lf": "kept", "breakout_lf": "kept",
        "trend_ens_lf": "improved", "breakout_slow_lf": "improved", "xs_top_lf": "new",
        "hold_btc": "benchmark", "basket_long": "benchmark", "main": "blend",
        "macd": "pruned", "breakout": "pruned", "xsmom_63": "pruned", "xs_rev_5": "pruned",
        "dl_lstm": "pruned", "dl_mom_trans": "pruned", "dl_deepm": "pruned",
        "trend_regime_lf": "rejected", "tsmom_slow_lf": "rejected", "trend_fund_lf": "rejected",
        "basket_regime": "rejected", "ml_gbm5_lf": "rejected"}
roster = []
for n, role in ROLE.items():
    dn = "ml_gbm5_lf_retrain1m" if n == "ml_gbm5_lf" else n
    d = dev.loc[dn] if dn in dev.index else None
    test = sim_ret.get(n)
    src = "simulation" if test is not None else ("backtest" if n in ref.index else None)
    if test is None and n in ref.index:
        test = float(ref.loc[n, "return"])
    roster.append({"name": n, "role": role,
                   "dev_a": None if d is None else float(d["sharpe_2021-23"]),
                   "dev_b": None if d is None else float(d["sharpe_2023-25"]),
                   "test": test, "src": src})

intra_dev = pd.read_csv(R / "dev_5min_policies.csv", index_col=0)
intra_int = pd.read_csv(R / "dev_5min_retrain_intervals.csv", index_col=0)
intra_test = pd.read_csv(S / "y1/ref2/reference_intra.csv", index_col=0)
mlg = json.load(open(S / "y1/ml_gross.json"))
ml_dev = dev[dev.index.str.startswith("ml_")]
prune = pd.read_csv(R / "pruning_rule_by_year.csv")

twins = acc[(acc.vault != "0") & (acc.sweeps > 0)].sort_values("vs_control", ascending=False)
data = {
    "t": [str(x.date()) for x in ew.index],
    "market": {"ew30": ew.round(2).tolist(), "btc": btc.round(2).tolist()},
    "curves": {a: s.tolist() for a, s in curves.items()},
    "accounts": acc[["account", "strategy", "vault", "total", "ret", "max_dd", "sweeps", "banked", "net_exp",
                     "vs_control", "fills", "fees", "spread", "funding"]].round(5).to_dict("records"),
    "roster": roster,
    "intra_dev": intra_dev.reset_index().rename(columns={"index": "name"}).round(4).to_dict("records"),
    "intra_int": intra_int.reset_index().rename(columns={"index": "name"}).round(4).to_dict("records"),
    "intra_test": intra_test.reset_index().rename(columns={"index": "name"}).round(4).to_dict("records"),
    "ml_test": mlg,
    "ml_dev": ml_dev.reset_index().rename(columns={"index": "name"}).round(4).to_dict("records"),
    "prune": prune.round(4).to_dict("records"),
    "twins": twins[["account", "sweeps", "banked", "total", "vs_control"]].round(2).to_dict("records"),
}
Path(S / "y1/page/data.json").write_text(json.dumps(data, default=float))
print("ok", len(data["t"]), "days", len(curves), "curves")
print(intra_test.round(3).to_string())
