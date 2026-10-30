"""vault-2w experiment: per-account results, vault vs control, market context, attribution, confidence.

usage: analyse.py <state_dir> <data_dir> <experiment.json> <out_dir>
state_dir holds intraday.sqlite and lab.sqlite as of the close-out; data_dir close_5m/close_1h parquet
(spot closes indexed by bar OPEN time, relabelled to close time here, as the engines do).
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tradinglab.intraday.live import IntradayEngine
from tradinglab.live.state import StateStore

STATE, DATA, META, OUT = (Path(p) for p in sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
meta = json.loads(META.read_text())
START = pd.Timestamp(meta["start"]).tz_convert("UTC")
END = pd.Timestamp(meta["end"]).tz_convert("UTC")
BASE = 100_000.0
RNG = np.random.default_rng(20261030)


def iso(s):
    return pd.to_datetime(s, utc=True, format="ISO8601")


def split(name: str):
    m = re.match(r"^(?:intraday:|shadow:)?(.*?)(?:\+vault(\d+))?$", name)
    return m.group(1), (int(m.group(2)) if m.group(2) else 0)


def max_dd(x: pd.Series) -> float:
    return float((x / x.cummax() - 1.0).min()) if len(x) else 0.0


def ols(r: pd.Series, m: pd.Series):
    df = pd.concat([r, m], axis=1).dropna()
    if len(df) < 10 or df.iloc[:, 1].var() == 0:
        return None
    y, x = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    beta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    alpha = y.mean() - beta * x.mean()
    resid = y - alpha - beta * x
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0.0
    se_a = resid.std(ddof=2) / np.sqrt(len(y))
    return {"beta": float(beta), "alpha_per_period": float(alpha), "alpha_t": float(alpha / se_a) if se_a > 0 else None,
            "r2": float(r2), "n": int(len(y))}


def block_bootstrap_total(r: np.ndarray, block: int, n: int = 4000) -> tuple[float, float, float]:
    """CI of the compounded total return by circular block bootstrap of period returns."""
    T = len(r)
    if T < block * 2:
        return (np.nan, np.nan, np.nan)
    nb = int(np.ceil(T / block))
    starts = RNG.integers(0, T, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
    sims = np.prod(1.0 + r[idx.reshape(n, -1)[:, :T]], axis=1) - 1.0
    return float(np.quantile(sims, 0.05)), float(np.quantile(sims, 0.5)), float(np.quantile(sims, 0.95))


# ------------------------------------------------------------------ market data
c5 = pd.read_parquet(DATA / "close_5m.parquet")
c5.index = c5.index + pd.Timedelta(minutes=5)
c1 = pd.read_parquet(DATA / "close_1h.parquet")
c1.index = c1.index + pd.Timedelta(hours=1)
win5 = c5[(c5.index >= START.floor("5min")) & (c5.index <= END)]
win1 = c1[(c1.index >= START.floor("h")) & (c1.index <= END)]          # hourly closes inside the window
c530 = pd.read_parquet(DATA / "close_5m30.parquet")
c530.index = c530.index + pd.Timedelta(minutes=5)
win530 = c530[(c530.index >= START.floor("5min")) & (c530.index <= END)]


def market_stats(close: pd.DataFrame, per_year: float) -> dict:
    r = close.pct_change(fill_method=None)
    ew = r.mean(axis=1, skipna=True).iloc[1:]
    ew_curve = (1 + ew).cumprod()
    first, last = close.iloc[0], close.iloc[-1]
    coin = (last / first - 1.0).sort_values()
    path = np.log(close.ffill()).diff().abs().sum()
    eff = (np.log(last / first).abs() / path)
    return {
        "ew_return": float(ew_curve.iloc[-1] - 1.0), "ew_max_dd": max_dd(ew_curve),
        "ew_vol_ann": float(ew.std() * np.sqrt(per_year)),
        "btc_return": float(close["BTCUSDT"].iloc[-1] / close["BTCUSDT"].iloc[0] - 1.0),
        "btc_max_dd": max_dd(close["BTCUSDT"]),
        "coin_returns": {k: float(v) for k, v in coin.items()},
        "median_efficiency_ratio": float(eff.median()),
        "up_coins": int((coin > 0).sum()), "n_coins": int(len(coin)),
    }


# both baskets on 5-minute closes from the last close before the start to the last close before the end
market = {"intraday_12": market_stats(win5, 365 * 288), "daily_30": market_stats(win530, 365 * 288)}
ew1 = win1.pct_change(fill_method=None).mean(axis=1, skipna=True)
btc_daily = win1["BTCUSDT"].resample("1D", origin=START.floor("h"), closed="right", label="right").last()
market["btc_path_daily"] = {str(k): float(v) for k, v in btc_daily.dropna().items()}
ew_curve_1h = (1 + ew1.fillna(0)).cumprod()
ew5 = win5.pct_change(fill_method=None).mean(axis=1, skipna=True)
ew_curve_5m_h = (1 + ew5.fillna(0)).cumprod().iloc[::12]       # every hour from the 23:45 start close
ew530 = win530.pct_change(fill_method=None).mean(axis=1, skipna=True)
ew30_curve_5m_h = (1 + ew530.fillna(0)).cumprod().iloc[::12]
btc_5m_h = (win5["BTCUSDT"] / win5["BTCUSDT"].iloc[0]).iloc[::12]


# ------------------------------------------------------------------ accounts
def book(db: str, kind: str):
    store = StateStore(STATE / db)
    con = sqlite3.connect(STATE / db)
    con.row_factory = sqlite3.Row
    accounts = [dict(r) for r in con.execute("SELECT * FROM accounts ORDER BY name")]
    rows, curves = [], {}
    for a in accounts:
        name = a["name"]
        fam, thr = split(name)
        eq = store.equity_curve(name)
        eq = eq[eq.index <= END + pd.Timedelta(minutes=5)]
        vault = pd.DataFrame([dict(r) for r in con.execute(
            "SELECT ts, amount, equity_marked, equity_realised, mark_ts FROM vault WHERE account=? ORDER BY id", (name,))])
        if not vault.empty:
            vault["mark"] = iso(vault["mark_ts"].fillna(vault["ts"]))
        # wealth = equity + profit banked at earlier marks (a sweep's own mark still holds the profit)
        banked = np.zeros(len(eq))
        for _, v in vault.iterrows():
            banked += np.where(eq.index > v["mark"], v["amount"], 0.0)
        wealth = eq["equity"] + banked
        orders = pd.DataFrame([dict(r) for r in con.execute(
            "SELECT * FROM orders WHERE account=? AND status='filled'", (name,))])
        fund = pd.DataFrame([dict(r) for r in con.execute(
            "SELECT ts, amount FROM funding WHERE account=?", (name,))], columns=["ts", "amount"])
        fees = float(orders["fee"].sum()) if len(orders) else 0.0
        slip = float(orders["slippage_cost"].fillna(0).sum()) if len(orders) else 0.0
        turnover = float((orders["qty"].abs() * orders["fill_price"]).sum()) if len(orders) else 0.0
        funding_paid = -float(fund["amount"].sum()) if len(fund) else 0.0
        final_eq = float(eq["equity"].iloc[-1])
        vtot = float(vault["amount"].sum()) if not vault.empty else 0.0
        w_final = final_eq + vtot
        hourly = wealth.resample("1h").last().dropna()
        hr = hourly.pct_change().dropna()
        w_end = wealth.copy()   # the close-out mark (seconds after the end) belongs to the last day
        w_end.index = w_end.index.where(w_end.index <= END, END)
        w_end = w_end.groupby(level=0).last()
        days = [START + pd.Timedelta(days=k) for k in range(1, 15)]   # 14 full days from the start
        daily = pd.Series([float(w_end[w_end.index <= d].iloc[-1]) for d in days], index=days)
        dr = pd.concat([pd.Series([BASE], index=[START]), daily]).pct_change().dropna()
        # regression on clock-hour buckets, the same buckets as the wealth series
        mkt = (1 + ew5.fillna(0)).cumprod().resample("1h").last() if kind == "intraday" else ew_curve_1h.resample("1h").last()
        reg = ols(hr, mkt.pct_change().reindex(hr.index))
        lo, med, hi = block_bootstrap_total(hr.to_numpy(), block=24)
        row = {
            "book": kind, "account": name, "strategy": fam, "vault_pct": thr, "strategy_desc": a.get("strategy"),
            "final_equity": final_eq, "vault": vtot, "wealth": w_final, "pnl": w_final - BASE,
            "return": w_final / BASE - 1.0, "sweeps": int(len(vault)),
            "first_sweep": str(vault["mark"].min()) if not vault.empty else None,
            "peak_wealth": float(wealth.max()), "max_dd": max_dd(pd.concat([pd.Series([BASE]), wealth])),
            "trades": int(len(orders)), "turnover_x": turnover / BASE, "fees": fees, "spread_impact": slip,
            "funding_paid": funding_paid, "costs": fees + slip + funding_paid,
            "gross_pnl_before_costs": w_final - BASE + fees + slip + funding_paid,
            # both engines store gross/net exposure as fractions of equity
            "avg_gross_exposure": float(eq["gross"].mean()), "avg_net_exposure": float(eq["net"].mean()),
            "time_in_market": float((eq["gross"] > 0.01).mean()),
            "daily_sharpe": float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else None,
            "hourly_sharpe_ann": float(hr.mean() / hr.std() * np.sqrt(365 * 24)) if hr.std() > 0 else None,
            "win_days": int((dr > 0).sum()), "n_days": int(len(dr)),
            "boot_total_p05": lo, "boot_total_p50": med, "boot_total_p95": hi,
            "regression": reg,
        }
        if kind == "intraday":
            ns = SimpleNamespace(store=store)
            ns._fills = lambda n, f, m, _ns=ns: IntradayEngine._fills(_ns, n, f, m)
            close = c5[(c5.index >= START - pd.Timedelta(days=1)) & (c5.index <= END + pd.Timedelta(minutes=5))]
            filled = store.orders(name, limit=10_000_000)
            filled = filled[filled["status"] == "filled"] if len(filled) else filled
            row["attribution"] = IntradayEngine._attribution(ns, name, filled, close)
        rows.append(row)
        curves[name] = {"t": [str(t) for t in hourly.index], "wealth": [round(float(x), 2) for x in hourly],
                        "equity": [round(float(x), 2) for x in eq["equity"].resample("1h").last().reindex(hourly.index)]}
        # audit: vault invariants
        if not vault.empty:
            row["audit_sweeps_at_threshold"] = bool((vault["equity_marked"] >= BASE * (1 + thr / 100) - 1e-6).all())
            row["audit_amounts_positive"] = bool((vault["amount"] > 0).all())
            after = [float(eq["equity"][eq.index > m].iloc[0]) if (eq.index > m).any() else None for m in vault["mark"]]
            row["audit_equity_after_sweep"] = after
    store.close()
    return rows, curves


ri, ci = book("intraday.sqlite", "intraday")
rd, cd = book("lab.sqlite", "daily")
rows = ri + rd
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("regression", "attribution", "audit_equity_after_sweep")}
                   for r in rows])

# vault twin vs its own no-vault control
cmp = []
for (bk, fam), g in df.groupby(["book", "strategy"]):
    ctrl = g[g["vault_pct"] == 0].iloc[0]
    for _, r in g[g["vault_pct"] > 0].iterrows():
        cmp.append({"book": bk, "strategy": fam, "vault_pct": int(r["vault_pct"]), "control_return": ctrl["return"],
                    "vault_return": r["return"], "delta_pnl": r["wealth"] - ctrl["wealth"], "vault": r["vault"],
                    "final_equity": r["final_equity"], "sweeps": int(r["sweeps"]),
                    "control_max_dd": ctrl["max_dd"], "vault_max_dd": r["max_dd"]})
cmp = pd.DataFrame(cmp)

# pre-sweep identity check: a twin and its control share signals, so equity must match until the twin's first sweep
ident = []
for book_rows, curves in ((ri, ci), (rd, cd)):
    byname = {r["account"]: r for r in book_rows}
    for r in book_rows:
        if r["vault_pct"] == 0:
            continue
        ctrl_name = r["account"].split("+vault")[0]
        a, b = pd.Series(curves[r["account"]]["equity"], index=curves[r["account"]]["t"]), \
            pd.Series(curves[ctrl_name]["equity"], index=curves[ctrl_name]["t"])
        cut = r["first_sweep"] or str(END)
        common = [t for t in a.index if t in b.index and pd.Timestamp(t) < pd.Timestamp(cut) - pd.Timedelta(hours=1)]
        diff = float((a[common] - b[common]).abs().max()) if common else 0.0
        ident.append({"account": r["account"], "max_abs_equity_diff_before_first_sweep": diff, "hours": len(common)})

# multiple-testing yardstick: best of N independent zero-skill accounts with the observed per-account noise
ctrl = df[df["vault_pct"] == 0]
sig = []
for _, r in ctrl.iterrows():
    c = (ci if r["book"] == "intraday" else cd)[r["account"]]
    w = pd.Series(c["wealth"]).pct_change().dropna()
    sig.append(w.std() * np.sqrt(len(w)))
sig = np.array(sig)
sims = RNG.normal(0, 1, size=(20000, len(sig))) * sig
yard = {"n_strategies": int(len(sig)), "expected_best_return_under_zero_skill": float(sims.max(axis=1).mean()),
        "p95_best_return_under_zero_skill": float(np.quantile(sims.max(axis=1), 0.95))}
# per strategy: z of its own 14-day log return against its own noise, and the chance that at least one of the
# 20 controls scores this high by luck alone (Sidak)
from math import erf, sqrt
per = []
for (_, r), s_ in zip(ctrl.iterrows(), sig):
    z = float(np.log1p(r["return"]) / s_) if s_ > 0 else 0.0
    p1 = 0.5 * (1 - erf(z / sqrt(2)))
    per.append({"account": r["account"], "return": float(r["return"]), "sigma_14d": float(s_), "z": z,
                "p_one_sided": p1, "p_any_of_20": 1 - (1 - p1) ** len(sig)})
yard["per_strategy"] = per

res = {"window": {"start": str(START), "end": str(END)}, "market": market, "accounts": rows,
       "vault_vs_control": cmp.to_dict("records"), "twin_identity": ident, "zero_skill_yardstick": yard,
       "totals": {
           "vault_all": float(df["vault"].sum()),
           "vault_by_book": df.groupby("book")["vault"].sum().to_dict(),
           "accounts_with_vault": int((df["vault"] > 0).sum()),
           "accounts_up": int((df["pnl"] > 0).sum()), "accounts": int(len(df)),
       }}
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
(OUT / "curves.json").write_text(json.dumps({"intraday": ci, "daily": cd,
                                             "market_ew12_h": {str(k): float(v) for k, v in ew_curve_5m_h.items()},
                                             "market_ew30_h": {str(k): float(v) for k, v in ew30_curve_5m_h.items()},
                                             "market_btc_h": {str(k): float(v) for k, v in btc_5m_h.items()}}))
df.to_csv(OUT / "accounts.csv", index=False)
cmp.to_csv(OUT / "vault_vs_control.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
print(json.dumps(market, indent=1)[:1500])
print(df[["book", "account", "final_equity", "vault", "pnl", "return", "sweeps", "max_dd", "trades", "fees",
          "spread_impact", "funding_paid", "daily_sharpe"]].round(4).to_string())
print(cmp.round(4).to_string())
print("identity", pd.DataFrame(ident)["max_abs_equity_diff_before_first_sweep"].describe())
print("yardstick", yard)
for r in ri:
    if r["vault_pct"] == 0:
        a = r["attribution"]
        print(r["account"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in (a or {}).items()})
