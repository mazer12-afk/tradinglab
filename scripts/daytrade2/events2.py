"""Event studies with futures metrics across the three development folds (forward moves in bp)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1])
from load import load_minutes
from tradinglab.config import load_config
from tradinglab.data.store import MarketData

MET = Path(sys.argv[2])
FOLDS = [("F1", "2022-10-01", "2023-08-01"), ("F2", "2023-08-01", "2024-09-01"), ("F3", "2024-09-01", "2025-10-29")]
m = load_minutes("2022-09-01", "2025-10-30")
c = m.close
syms = list(c.columns)
lc = np.log(c.ffill())
# 5-minute grid: label T = the minute after a 5-minute close (the metrics snapshot at T plus one minute of latency)
grid = c.index[(c.index.minute % 5 == 0)]
L = lc.reindex(grid)                                  # log close of the minute labelled T (closes at T+1m)
prev5 = lc.shift(1).reindex(grid)                     # close at T (the 5-minute bar close)
r5 = (lc.shift(1) - lc.shift(6)).reindex(grid)        # the 5-minute bar ending at T
sig = r5.ewm(span=288, min_periods=100).std().shift(1)
z = r5 / sig
fwd = {h: (lc.shift(-h).reindex(grid) - L) * 1e4 for h in (15, 60, 240)}
met = {s: pd.read_parquet(MET / f"{s}.parquet") for s in syms}
oi = pd.DataFrame({s: met[s]["sum_open_interest_value"] for s in syms}).reindex(grid)
doi = np.log(oi).diff()                               # OI change over the 5 minutes ending at T
doi_z = doi / doi.ewm(span=288, min_periods=100).std().shift(1)
taker = pd.DataFrame({s: met[s]["sum_taker_long_short_vol_ratio"] for s in syms}).reindex(grid)
tk = np.log(taker).rolling(12, min_periods=10).mean()  # last hour of taker buy/sell imbalance
tk_z = (tk - tk.rolling(288 * 7, min_periods=288).mean()) / tk.rolling(288 * 7, min_periods=288).std()
rows = []


def study(name, mask, sign):
    for f, a, b in FOLDS:
        sel = mask & pd.DataFrame(np.repeat(((grid >= pd.Timestamp(a, tz="UTC")) & (grid < pd.Timestamp(b, tz="UTC")))[:, None], len(syms), axis=1), index=grid, columns=syms)
        row = {"idea": name, "fold": f, "events": int(sel.sum().sum())}
        for h, F in fwd.items():
            x = (F * sign)[sel].stack()
            row[f"{h}m"] = round(float(x.mean()), 1) if len(x) else np.nan
        rows.append(row)


for k in (4.0, 5.0):
    crash, spike = z < -k, z > k
    study(f"buy crash |z|>{k:g}, OI falling (z<-2)", crash & (doi_z < -2), 1)
    study(f"buy crash |z|>{k:g}, OI not falling", crash & (doi_z >= -2), 1)
    study(f"fade spike |z|>{k:g}, OI falling", spike & (doi_z < -2), -1)
    study(f"fade spike |z|>{k:g}, OI not falling", spike & (doi_z >= -2), -1)
on_hour = pd.DataFrame(np.repeat((grid.minute == 0)[:, None], len(syms), axis=1), index=grid, columns=syms)
study("fade 1h taker buying (z>2.5)", on_hour & (tk_z > 2.5), -1)
study("fade 1h taker selling (z<-2.5)", on_hour & (tk_z < -2.5), 1)
# funding settlements: sign of the last settled rate → the move over the hour before and after the next one
cfg = load_config(); md = MarketData(cfg.paths.data_dir)
fr = pd.DataFrame({s: md.funding_8h(s, pd.Timestamp("2022-08-01")) for s in syms})
fr.index = fr.index.round("min")
last = fr.reindex(grid, method="ffill")
settle = pd.DataFrame(np.repeat((grid.hour.isin([0, 8, 16]) & (grid.minute == 0))[:, None], len(syms), axis=1), index=grid, columns=syms)
before = settle.shift(-12, fill_value=False)          # one hour before a settlement
pos = last > 0.0002
neg = last < 0
study("short 1h into settlement when funding > 2bp", before & pos, -1)
study("long 1h into settlement when funding < 0", before & neg, 1)
study("long 1h after settlement when funding > 2bp", settle & pos, 1)
t = pd.DataFrame(rows)
pd.set_option("display.width", 220, "display.max_rows", 200)
print(t.pivot(index="idea", columns="fold", values=["events", "60m", "240m"]).to_string())
t.to_csv(sys.argv[3], index=False)
