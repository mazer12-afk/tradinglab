"""Event studies on the design period (Aug 2023 → Aug 2024): average move after each signal, in bp."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1])
from load import load_minutes

A0, A1 = "2023-08-01", "2024-09-01"
m = load_minutes("2023-06-15", A1)
c = m.close
print("minutes", c.shape, "missing", float(c.isna().mean().mean()), flush=True)
lc = np.log(c.ffill())
H = [5, 15, 30, 60, 120, 240]
fwd = {h: (lc.shift(-h) - lc) * 1e4 for h in H}          # bp, from the close of the decision minute
in_a_1d = np.asarray(c.index >= pd.Timestamp(A0, tz="UTC"))
in_a = pd.DataFrame(np.repeat(in_a_1d[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)


def summarise(name, mask, sign):
    """mean signed forward move (bp) and t-stat over events in mask (DataFrame of bool)"""
    out = {"events": int(mask[in_a].sum().sum())}
    for h in H:
        x = (fwd[h] * sign)[mask & in_a].stack()
        out[f"{h}m"] = round(float(x.mean()), 1)
        out[f"t{h}"] = round(float(x.mean() / x.std() * np.sqrt(len(x))), 1) if len(x) > 2 else np.nan
    rows.append(pd.Series(out, name=name))


rows = []
# 1. extreme 5-minute moves (decided on 5-minute closes), reversal sign
r5 = (lc - lc.shift(5)) * 1e4
on5 = pd.DataFrame(np.repeat((np.arange(len(c)) % 5 == 4)[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
vol5 = r5.where(on5).ewm(span=288, min_periods=100, ignore_na=True).std().shift(5).ffill()
z5 = r5 / vol5
mkt = r5.mean(axis=1)
idio = r5.sub(mkt, axis=0)
zi = idio / vol5
for k in (3, 4, 5):
    summarise(f"reversal after |5m z| > {k}", on5 & (z5.abs() > k), -np.sign(z5))
    summarise(f"reversal after |5m idio z| > {k}", on5 & (zi.abs() > k), -np.sign(zi))
# 6. BTC lead: BTC's last-minute move > 3 sigma, alt did not move yet → alt follows BTC?
r1 = (lc - lc.shift(1)) * 1e4
v1 = r1.ewm(span=1440, min_periods=500).std().shift(1)
zb = (r1["BTCUSDT"] / v1["BTCUSDT"])
alt_quiet = (r1.abs() < v1)
lead = pd.DataFrame(np.repeat((zb.abs() > 3).to_numpy()[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
lead = lead & alt_quiet
lead["BTCUSDT"] = False
summarise("alts follow a >3 sigma BTC minute", lead, pd.DataFrame(np.repeat(np.sign(zb).to_numpy()[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns))
# 4. breakouts of the prior 4h / 24h high or low on 5-minute closes, continuation sign
for n, lab in ((48, "4h"), (288, "24h")):
    hi = c.rolling(5 * n, min_periods=5 * n).max().shift(5)
    lo = c.rolling(5 * n, min_periods=5 * n).min().shift(5)
    up = on5 & (c > hi) & (c.shift(5) <= hi)
    dn = on5 & (c < lo) & (c.shift(5) >= lo)
    summarise(f"continuation after {lab} breakout", up | dn, up.astype(float) - dn.astype(float))
# 2. late-day momentum: sign of the UTC day's return up to tau predicts tau → close of the day
day = c.index.floor("D")
for hh, mm in ((20, 0), (22, 0), (23, 0)):
    at = (c.index.hour == hh) & (c.index.minute == mm)
    open_px = c.groupby(day).transform("first")
    sig = np.sign(lc - np.log(open_px))
    mask = pd.DataFrame(np.repeat(at[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    summarise(f"day-so-far momentum at {hh:02d}:{mm:02d}", mask, sig)
# 5. funding settlements: the last settled rate's sign → the move into/after the next settlement
print(pd.DataFrame(rows).to_string(), flush=True)
if len(sys.argv) > 2:
    pd.DataFrame(rows).to_csv(sys.argv[2])

# 3. hour-of-day: mean 1h return of the equal-weight basket and BTC per UTC hour, A period
r60 = (lc.shift(-60) - lc) * 1e4
top = (c.index.minute == 0) & in_a_1d
bask = r60[top].mean(axis=1)
btc = r60["BTCUSDT"][top]
hod = pd.DataFrame({"basket_bp": bask.groupby(bask.index.hour).mean(), "basket_t": bask.groupby(bask.index.hour).apply(lambda x: x.mean() / x.std() * np.sqrt(len(x))),
                    "btc_bp": btc.groupby(btc.index.hour).mean(), "btc_t": btc.groupby(btc.index.hour).apply(lambda x: x.mean() / x.std() * np.sqrt(len(x)))})
print(hod.round(1).T.to_string())
if len(sys.argv) > 2:
    hod.to_csv(sys.argv[2].replace('.csv', '_hour_of_day.csv'))
