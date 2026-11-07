"""Forward (paper) test of the frozen funding-carry rules R2, R3 and R4 on live Binance data (PROTOCOL.md section 5).

Every day it rebuilds the panel from www.binance.com (spot and perpetual hourly candles, funding settlements),
applies the frozen rules (research/proof/carry.py, confirm_carry.py) from START to the last complete day, and writes
the records. Where Binance's archive already covers a day, the same book is rebuilt from the archive and the two
are compared: they must agree (the implementation check).

    python research/proof/carry_forward.py --records records
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import carry as C  # noqa: E402
import confirm_carry as K  # noqa: E402

API = "https://www.binance.com"
START = pd.Timestamp("2026-11-08", tz="UTC")       # first day after the rules were frozen and confirmed
ENDS = START + pd.Timedelta(days=28)               # minimum length; it keeps running afterwards
FRESH = json.loads((HERE / "fresh_universe.json").read_text())["symbols"]
UNIVERSE = C.TWELVE + C.OTHERS + FRESH            # R2/R3 trade the 12 coins; R4 the whole wide universe


class LiveSource:
    """The same data as ArchiveSource, from Binance's live API."""

    def __init__(self):
        import requests

        self.s = requests.Session()

    def _get(self, path, params, tries=5):
        for k in range(tries):
            try:
                r = self.s.get(API + path, params=params, timeout=20)
                if r.status_code == 200:
                    return r.json()
            except Exception:                      # noqa: BLE001 - retried
                pass
            time.sleep(2 * (k + 1))
        raise ConnectionError(path)

    def _klines(self, path, s, start, limit):
        rows, t = [], int(start.value // 10**6)
        end = int(pd.Timestamp.now(tz="UTC").floor("h").value // 10**6)
        while t < end:
            r = self._get(path, {"symbol": s, "interval": "1h", "startTime": t, "endTime": end - 1, "limit": limit})
            if not r:
                break
            rows += r
            t = int(r[-1][0]) + 3_600_000
            time.sleep(0.1)
        if not rows:
            return pd.Series(dtype=float)
        x = pd.Series([float(r[4]) for r in rows], index=pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True))
        return x[~x.index.duplicated()]

    def spot(self, s, start):
        return self._klines("/api/v3/klines", s, start, 1000)

    def perp(self, s, start):
        return self._klines("/fapi/v1/klines", s, start, 1500)

    def funding(self, s, start):
        rows, t = [], int(start.value // 10**6)
        while True:
            r = self._get("/fapi/v1/fundingRate", {"symbol": s, "startTime": t, "limit": 1000})
            rows += r
            if len(r) < 1000:
                break
            t = int(r[-1]["fundingTime"]) + 1
        return pd.Series({pd.Timestamp(int(x["fundingTime"]), unit="ms", tz="UTC"): float(x["fundingRate"]) for x in rows}, dtype=float)


def subset(p, syms):
    """A view of a CarryPanel restricted to ``syms`` (the rules normalise across the coins they see)."""
    import copy

    q = copy.copy(p)
    q.symbols = [x for x in p.symbols if x in syms]
    for k in ("spot_ret", "perp_ret", "fund_next", "live"):
        setattr(q, k, getattr(p, k)[q.symbols])
    q.fund_hist = {x: p.fund_hist[x] for x in q.symbols}
    return q


def rule_weights(p, rule):
    if rule == "R4":
        return C.funding_carry(p, **K.R4)
    return C.funding_carry(p, **K.R2) if rule == "R2" else K.btc_eth(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="records")
    a = ap.parse_args()
    rec = Path(a.records)
    rec.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").floor("D")
    last = today - pd.Timedelta(days=1)           # the last day whose next-day return is complete
    if last < START:
        print("nothing to do before", START.date())
        return
    end = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    # the rules need 14 days of funding history and the panel 30 days of listing; start the panel early
    lo = (START - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    live = C.CarryPanel(UNIVERSE, lo, end, source=LiveSource())
    status = {"started": str(START.date()), "minimum_end": str(ENDS.date()), "through": str(last.date()),
              "updated": str(pd.Timestamp.now(tz="UTC")), "rules": {}}
    daily = {}
    arch_panel = None
    for rule, syms in (("R2", C.TWELVE), ("R3", C.TWELVE), ("R4", UNIVERSE)):
        p = subset(live, syms)
        W = rule_weights(p, rule)
        b = C.book(p, W)
        b = b[b.index >= START]
        eq = (1 + b["net"]).cumprod()
        held = W.iloc[-1][W.iloc[-1] > 0]
        status["rules"][rule] = {"days": int(len(b)), "return": round(float(eq.iloc[-1] - 1), 6),
                                 "annualised": round(float(eq.iloc[-1] ** (365 / max(len(b), 1)) - 1), 4),
                                 "funding": round(float(b["funding"].sum()), 6), "basis": round(float(b["basis"].sum()), 6),
                                 "costs": round(float(b["cost"].sum()), 6), "worst_drawdown": round(float((eq / eq.cummax() - 1).min()), 6),
                                 "holding": {k: round(float(v), 3) for k, v in held.items()}}
        daily[rule] = b.assign(equity=eq)
        # implementation check against the archive wherever it already has the data
        try:
            arch_panel = arch_panel or C.CarryPanel(UNIVERSE, lo, end)
            arch = subset(arch_panel, syms)
            Wa = rule_weights(arch, rule)
            ba = C.book(arch, Wa)
            common = b.index.intersection(ba.index[ba[["basis", "funding"]].abs().sum(axis=1) > 0])
            if len(common):
                d = (b.loc[common, "net"] - ba.loc[common, "net"]).abs()
                pos = (W.loc[common] - Wa.reindex(columns=W.columns).fillna(0.0).loc[common]).abs().max().max()
                status["rules"][rule]["archive_check"] = {"days": int(len(common)), "max_daily_net_diff": round(float(d.max()), 8),
                                                          "max_weight_diff": round(float(pos), 6)}
        except Exception as exc:                   # noqa: BLE001 - the archive lags; report and carry on
            status["rules"][rule]["archive_check"] = {"error": str(exc)[:200]}
    for rule, d in daily.items():
        d.to_csv(rec / f"{rule}_daily.csv")
    (rec / "status.json").write_text(json.dumps(status, indent=1) + "\n")
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
