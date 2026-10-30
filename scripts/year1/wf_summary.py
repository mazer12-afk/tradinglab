"""Walk-forward simulation summary: per strategy monthly returns, vault twins vs control, market."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run = Path(sys.argv[1])
BASE = 100_000.0


def load(db):
    c = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
    e = pd.read_sql("select account, ts, equity, net, gross from equity", c)
    e["ts"] = pd.to_datetime(e.ts, utc=True, format="ISO8601")
    v = pd.read_sql("select account, amount, mark_ts from vault", c)
    v["mark_ts"] = pd.to_datetime(v.mark_ts, utc=True, format="ISO8601")
    return e, v


def wealth(e, v, acct):
    """Equity + vault per mark (a sweep's transfer counts from its mark on)."""
    g = e[e.account == acct].set_index("ts").sort_index()
    w = g["equity"].copy()
    vv = v[v.account == acct]
    for _, r in vv.iterrows():
        w[w.index > r.mark_ts] += r.amount
    return w, g


rows, monthly = [], {}
for db in ("intraday.sqlite", "lab.sqlite"):
    if not (run / db).exists():
        continue
    e, v = load(run / db)
    for acct in sorted(e.account.unique()):
        w, g = wealth(e, v, acct)
        base = acct.split("+vault")[0]
        dd = (w / w.cummax() - 1).min()
        rows.append({"book": "5-min" if db.startswith("intra") else "daily", "account": acct, "strategy": base,
                     "vault": acct.split("+vault")[1] if "+vault" in acct else "0",
                     "total": w.iloc[-1], "ret": w.iloc[-1] / BASE - 1, "max_dd": dd,
                     "sweeps": int((v.account == acct).sum()), "banked": float(v[v.account == acct].amount.sum()),
                     "net_exp": float(g["net"].mean())})
        if "+vault" not in acct:
            m = w.resample("ME").last()
            prev = pd.concat([pd.Series([BASE], index=[w.index[0]]), m.iloc[:-1]])
            monthly[acct] = (m.to_numpy() / prev.to_numpy() - 1)
            monthly_index = m.index
t = pd.DataFrame(rows)
ctrl = t[t.vault == "0"].set_index("account")
t["vs_control"] = t.apply(lambda r: r.total - ctrl.loc[r.strategy, "total"], axis=1)
pd.set_option("display.width", 220)
print("== controls")
c = t[t.vault == "0"].sort_values("ret", ascending=False)
mm = pd.DataFrame(monthly, index=[f"{x:%b}" for x in monthly_index]).T
print(c.set_index("account")[["book", "ret", "max_dd", "net_exp"]].join(mm).to_string(float_format=lambda x: f"{x:+.2%}" if abs(x) < 5 else f"{x:,.0f}"))
print("\n== vault twins that swept")
tw = t[(t.vault != "0") & (t.sweeps > 0)].sort_values("vs_control", ascending=False)
print(tw[["account", "sweeps", "banked", "ret", "vs_control"]].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print("\ntwins that swept: %d; total vs controls %+.0f; positive %d" % (len(tw), tw.vs_control.sum(), (tw.vs_control > 0).sum()))
t.to_csv(run / "wf_summary.csv", index=False)
