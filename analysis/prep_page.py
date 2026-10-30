import json, sys
import numpy as np
import pandas as pd

O = "out/"
r = json.load(open(O + "results.json"))
cv = json.load(open(O + "curves.json"))
c1 = pd.read_parquet("data/close_1h.parquet"); c1.index = c1.index + pd.Timedelta(hours=1)
start, end = pd.Timestamp(r["window"]["start"]), pd.Timestamp(r["window"]["end"])

def hourly_index(d):
    s = pd.Series(d); s.index = pd.to_datetime(s.index, utc=True)
    # 5-minute closes sampled every hour from the 23:45 close before the start to the 23:45 close before the end
    return s / s.iloc[0] * 100

ew12 = hourly_index(cv["market_ew12_h"]); ew30 = hourly_index(cv["market_ew30_h"])
btc = hourly_index(cv["market_btc_h"])
mk = pd.DataFrame({"ew12": ew12, "ew30": ew30, "btc": btc}).dropna()
market = {"t": [t.strftime("%Y-%m-%dT%H:%MZ") for t in mk.index], **{k: [round(float(x), 3) for x in mk[k]] for k in mk}}

def num(x, nd=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)

acc = []
for a in r["accounts"]:
    reg = a.get("regression") or {}
    at = a.get("attribution") or {}
    acc.append({
        "book": a["book"], "account": a["account"], "strategy": a["strategy"], "vault_pct": a["vault_pct"],
        "final_equity": num(a["final_equity"]), "vault": num(a["vault"]), "wealth": num(a["wealth"]), "pnl": num(a["pnl"]),
        "ret": num(a["return"], 5), "sweeps": a["sweeps"], "max_dd": num(a["max_dd"], 5), "trades": a["trades"],
        "turnover_x": num(a["turnover_x"], 2), "fees": num(a["fees"]), "spread": num(a["spread_impact"]),
        "funding": num(a["funding_paid"]), "costs": num(a["costs"]), "gross_pnl": num(a["gross_pnl_before_costs"]),
        "net_exp": num(a.get("avg_net_exposure"), 3), "gross_exp": num(a.get("avg_gross_exposure"), 3),
        "beta": num(reg.get("beta"), 3), "alpha_t": num(reg.get("alpha_t"), 2), "r2": num(reg.get("r2"), 3),
        "sharpe": num(a.get("daily_sharpe"), 2), "win_days": a.get("win_days"), "n_days": a.get("n_days"),
        "boot_lo": num(a.get("boot_total_p05"), 4), "boot_hi": num(a.get("boot_total_p95"), 4),
        "attr": {k: num(at.get(k)) for k in ("passive_pnl", "timing_pnl", "selection_pnl", "costs")} if at else None,
        "first_sweep": a.get("first_sweep"),
    })

def curve(book, name):
    c = cv[book][name]
    return {"t": [pd.Timestamp(t).strftime("%Y-%m-%dT%H:%MZ") for t in c["t"]], "w": c["wealth"]}

fams = {"lgbm_4h": ("intraday", "intraday:lgbm_4h"), "dl_deepm": ("daily", "shadow:dl_deepm"),
        "dl_lstm": ("daily", "shadow:dl_lstm"), "breakout": ("daily", "shadow:breakout")}
curves = {f: {str(th): curve(bk, base + (f"+vault{th}" if th else "")) for th in (0, 1, 2, 4)} for f, (bk, base) in fams.items()}
sweeps = []
import sqlite3
for db in ("intraday", "lab"):
    c = sqlite3.connect(f"state/{db}.sqlite")
    for row in c.execute("select account, mark_ts, amount, equity_marked from vault order by mark_ts"):
        sweeps.append({"account": row[0], "t": row[1], "amount": round(row[2], 2), "equity_marked": round(row[3], 2)})
out = {"window": r["window"], "market": market, "market_stats": {k: {kk: vv for kk, vv in v.items() if kk != "coin_returns"} for k, v in r["market"].items() if k != "btc_path_daily"},
       "coins12": r["market"]["intraday_12"]["coin_returns"], "accounts": acc, "curves": curves, "sweeps": sweeps,
       "yard": r["zero_skill_yardstick"], "totals": r["totals"]}
json.dump(out, open(O + "page_data.json", "w"), separators=(",", ":"))
print(len(json.dumps(out)) / 1e3, "KB")
