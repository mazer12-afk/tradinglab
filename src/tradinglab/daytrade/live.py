"""Real-time paper trading of the day-trading candidate on live Binance USD-M data.

The rule is the frozen candidate (config/daytrade2_candidate.json, with the timing fix) and its arms
(config/daytrade_forward_arms.json): at each 5-minute close, extreme moves are scored by the month's model
(trained on archive data before the month) and the arms buy the ones whose 4-hour forecast clears their
threshold, then sell when the holding time is up.

Live data comes from www.binance.com (fapi.binance.com is geo-blocked from GitHub's runners; the same data is
served there). Futures metrics are converted to the archive's labelling, which the model was trained on: the
live open-interest and long/short values stamped X are the archive's values stamped X - 5 minutes, and the
taker ratio carries the same stamp in both. With the timing fix, the decision at T + 1 minute needs values
stamped T - 10 minutes or earlier (archive labelling), which Binance has published by then.

Fills: market buy at the live ask plus ``impact``, market sell at the live bid less ``impact``, 5 bp taker fee
each way; funding at each settlement held through. Sizing: 20 % of realised equity per trade, one position per
coin, open notional at most ``max_gross`` × equity, the strongest forecast first. Everything is kept in the
records directory (its own git branch): state.json, trades.csv, decisions.csv, equity.csv, models/.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.daytrade import events as E
from tradinglab.daytrade.sim import Minutes

log = logging.getLogger(__name__)
API = "https://www.binance.com"
# live endpoint → [(archive column, live field)], minutes to add to the live stamp to get the archive label
METRICS = {
    "openInterestHist": ([("sum_open_interest", "sumOpenInterest"), ("sum_open_interest_value", "sumOpenInterestValue")], -5),
    "topLongShortPositionRatio": ([("sum_toptrader_long_short_ratio", "longShortRatio")], -5),
    "globalLongShortAccountRatio": ([("count_long_short_ratio", "longShortRatio")], -5),
    "takerlongshortRatio": ([("sum_taker_long_short_vol_ratio", "buySellRatio")], 0),
}
TAKER_BPS, IMPACT_BPS = 5.0, 1.0
BUFFER_DAYS = 21
MIN = pd.Timedelta(minutes=1)
FIVE = pd.Timedelta(minutes=5)
BAR_SETTLE = pd.Timedelta(seconds=3)       # a closed minute bar is read this long after its close
PUBLISHED = pd.Timedelta(minutes=8)        # an archive-labelled metrics row is fully published by stamp + 8 min
MET_POLL = pd.Timedelta(seconds=20)        # metrics are re-polled at most this often
MET_LOOKBACK = pd.Timedelta(hours=2)       # missing or partial metrics rows this recent are re-fetched
DECIDE_WAIT = pd.Timedelta(seconds=90)     # a decision waits at most this long past T + 1 minute for metrics
MAX_FAILURES = 120                         # consecutive failed cycles (~10 minutes) end the shift


# ----------------------------------------------------------------------------------------------- live API
class BinanceLive:
    """Public Binance USD-M market data via www.binance.com."""

    def __init__(self, session=None, base: str = API):
        import requests

        self.s = session or requests.Session()
        self.base = base

    def _get(self, path: str, params: dict, tries: int = 4):
        for k in range(tries):
            try:
                r = self.s.get(self.base + path, params=params, timeout=15)
                if r.status_code == 200:
                    return r.json()
                log.warning("%s %s → HTTP %s", path, params.get("symbol", ""), r.status_code)
            except Exception as exc:  # noqa: BLE001 - network hiccups are retried
                log.warning("%s failed: %s", path, exc)
            time.sleep(1.5 * (k + 1))
        raise ConnectionError(f"{path} unavailable")

    def server_time(self) -> pd.Timestamp:
        return pd.Timestamp(int(self._get("/fapi/v1/time", {})["serverTime"]), unit="ms", tz="UTC")

    def klines(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Closed 1-minute bars with open time in [start, end)."""
        rows, t = [], int(start.value // 10**6)
        stop = int(end.value // 10**6)
        while t < stop:
            r = self._get("/fapi/v1/klines", {"symbol": symbol, "interval": "1m", "startTime": t, "endTime": stop - 1,
                                              "limit": 1500})
            if not r:
                break
            rows += r
            t = int(r[-1][0]) + 60_000
            if len(r) < 1500:
                break
            time.sleep(0.25)                                    # stay well inside the request-weight limit
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype="float64")
        df = pd.DataFrame([x[:6] for x in rows], columns=["t", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("t").astype("int64"), unit="ms", utc=True)
        df = df.astype("float64")
        return df[~df.index.duplicated(keep="last")]

    def metrics(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.DataFrame:
        """5-minute futures metrics in archive labelling, archive labels in [start, end] (last 30 days only).

        These endpoints ignore ``startTime`` unless ``endTime`` is given too (they then return the latest rows), so
        the range is walked in windows of 500 stamps."""
        end = end if end is not None else pd.Timestamp.now(tz="UTC")
        cols = {}
        for ep, (fields, shift) in METRICS.items():
            rows = []
            t = int((start - pd.Timedelta(minutes=shift)).value // 10**6)
            stop = int((end - pd.Timedelta(minutes=shift)).value // 10**6)
            while t <= stop:
                hi = min(t + 499 * 300_000, stop)
                rows += self._get(f"/futures/data/{ep}", {"symbol": symbol, "period": "5m", "limit": 500,
                                                          "startTime": t, "endTime": hi})
                t = hi + 1
                if t <= stop:
                    time.sleep(0.2)
            idx = pd.to_datetime([int(x["timestamp"]) for x in rows], unit="ms", utc=True) + pd.Timedelta(minutes=shift)
            for col, f in fields:
                cols[col] = pd.Series([float(x[f]) for x in rows], index=idx, dtype="float64")
        df = pd.DataFrame({c: v[~v.index.duplicated(keep="last")] for c, v in cols.items()},
                          columns=[c for f, _ in METRICS.values() for c, _ in f]).sort_index()
        df.index.name = "create_time"
        return df[(df.index >= start) & (df.index <= end)]

    def book(self) -> dict[str, tuple[float, float]]:
        r = self._get("/fapi/v1/ticker/bookTicker", {})
        return {x["symbol"]: (float(x["bidPrice"]), float(x["askPrice"])) for x in r}

    def funding(self, symbol: str, start: pd.Timestamp) -> pd.Series:
        r = self._get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": int(start.value // 10**6), "limit": 1000})
        return pd.Series({pd.Timestamp(int(x["fundingTime"]), unit="ms", tz="UTC").round("min"): float(x["fundingRate"])
                          for x in r}, dtype="float64")


# --------------------------------------------------------------------------------------------- data buffer
class Buffer:
    """The last ``BUFFER_DAYS`` of 1-minute bars and futures metrics, topped up from the live API."""

    def __init__(self, api, symbols: list[str], cache: Path | None = None, days: int = BUFFER_DAYS):
        self.api, self.symbols, self.cache, self.days = api, list(symbols), cache, days
        self.bars: dict[str, pd.DataFrame] = {}
        self.met: dict[str, pd.DataFrame] = {}
        self.fund: dict[str, pd.Series] = {}

    def load(self, now: pd.Timestamp) -> None:
        lo = (now - pd.Timedelta(days=self.days)).floor("min")
        if self.cache and (self.cache / "bars.parquet").exists():
            b = pd.read_parquet(self.cache / "bars.parquet")
            m = pd.read_parquet(self.cache / "metrics.parquet")
            self.bars = {s: g.droplevel(0) for s, g in b.groupby(level=0)} if len(b) else {}
            self.met = {s: g.droplevel(0) for s, g in m.groupby(level=0)} if len(m) else {}
            log.info("buffer restored from cache")
        for s in self.symbols:
            b, m = self.bars.get(s, pd.DataFrame()), self.met.get(s, pd.DataFrame())
            if b.empty or b.index[-1] < lo:
                b = self.api.klines(s, lo, now.floor("min"))
            elif b.index[0] > lo + pd.Timedelta(hours=1):          # a short cached buffer: fill in its start
                b = pd.concat([self.api.klines(s, lo, b.index[0]), b])
            if m.empty or m.index[-1] < lo:
                m = self.api.metrics(s, lo.floor("5min"), now)
            elif m.index[0] > lo + pd.Timedelta(hours=1):
                m = pd.concat([self.api.metrics(s, lo.floor("5min"), m.index[0] - FIVE), m])
            self.bars[s], self.met[s] = b[~b.index.duplicated(keep="last")], m[~m.index.duplicated(keep="last")]
            self.fund[s] = self.api.funding(s, lo - pd.Timedelta(days=30))
        self._met_next = None
        self.update(now)

    def met_gap(self, s: str, now: pd.Timestamp):
        """The first archive label from which this coin's metrics should be (re)fetched, or None: labels known by
        now (stamp + PUBLISHED) that are missing or only partly published, looking back MET_LOOKBACK."""
        due = (now - PUBLISHED).floor("5min")
        m = self.met.get(s)
        if m is None or m.empty:
            return (now - pd.Timedelta(days=self.days)).floor("5min")
        if m.index[-1] < due - MET_LOOKBACK:
            return m.index[-1] + FIVE
        grid = pd.date_range(due - MET_LOOKBACK, due, freq="5min")
        bad = m.reindex(grid).isna().any(axis=1)
        return grid[bad.to_numpy()][0] if bad.any() else None

    def update(self, now: pd.Timestamp) -> None:
        end = (now - BAR_SETTLE).floor("min")                    # bars that closed a few seconds ago or earlier
        lo = end - pd.Timedelta(days=self.days)
        for s in self.symbols:
            b = self.bars.get(s, pd.DataFrame())
            start = b.index[-1] + MIN if len(b) else lo
            if start < end:
                add = self.api.klines(s, start, end)
                b = pd.concat([b, add]) if len(b) else add
            self.bars[s] = b[b.index >= lo]
        if getattr(self, "_met_next", None) is not None and now < self._met_next:
            return
        self._met_next = now + MET_POLL
        for s in self.symbols:
            m = self.met.get(s, pd.DataFrame())
            gap = self.met_gap(s, now)
            if gap is not None:
                add = self.api.metrics(s, gap, now)
                m = add.combine_first(m).sort_index() if len(m) else add   # fresh values fill the holes
            self.met[s] = m[m.index >= lo.floor("5min")]

    def complete(self, s: str, label: pd.Timestamp) -> bool:
        m = self.met.get(s)
        return m is not None and label in m.index and bool(m.loc[label].notna().all())

    def refresh_funding(self, now: pd.Timestamp) -> None:
        for s in self.symbols:
            self.fund[s] = self.api.funding(s, now - pd.Timedelta(days=60))

    def save(self) -> None:
        if not self.cache:
            return
        self.cache.mkdir(parents=True, exist_ok=True)
        pd.concat(self.bars, names=["symbol"]).to_parquet(self.cache / "bars.parquet")
        pd.concat(self.met, names=["symbol"]).to_parquet(self.cache / "metrics.parquet")

    def minutes(self) -> Minutes:
        idx = pd.date_range(min(b.index[0] for b in self.bars.values()), max(b.index[-1] for b in self.bars.values()),
                            freq="1min")
        f = {c: pd.DataFrame({s: self.bars[s][c] for s in self.symbols}).reindex(idx) for c in ("open", "high", "low", "close")}
        return Minutes(f["open"], f["high"], f["low"], f["close"])


# ---------------------------------------------------------------------------------------------- the engine
@dataclass
class Position:
    arm: str
    symbol: str
    decided: str
    entry_time: str
    entry_px: float
    notional: float
    exit_due: str
    pred: float


@dataclass
class ArmState:
    name: str
    theta: float
    hold: int
    size: float
    max_gross: float
    equity: float = 100_000.0                 # realised
    open: list = field(default_factory=list)


class LiveEngine:
    def __init__(self, rec: Path, symbols: list[str], api, models_dir: Path | None = None, arms: list[dict] | None = None,
                 half_spread: dict | None = None):
        self.rec = Path(rec)
        self.rec.mkdir(parents=True, exist_ok=True)
        self.symbols, self.api = list(symbols), api
        self.models_dir = Path(models_dir or self.rec / "models")
        st = self._read_state()
        arms = arms if arms is not None else json.loads((Path(__file__).resolve().parents[3] / "config" /
                                                         "daytrade_forward_arms.json").read_text())["arms"]
        self.arms = {}
        for a in arms:
            s = st.get("arms", {}).get(a["name"], {})
            self.arms[a["name"]] = ArmState(a["name"], a["theta"], a["hold"], a["size"], a["max_gross"],
                                            s.get("equity", 100_000.0), [Position(**p) for p in s.get("open", [])])
        self.state = st
        self.half_spread = half_spread or {}
        self.model = None
        self.model_month = None

    # ---------------------------------------------------------------- persistence
    def _read_state(self) -> dict:
        p = self.rec / "state.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def save(self) -> None:
        self.state["arms"] = {n: {"equity": a.equity, "open": [p.__dict__ for p in a.open]} for n, a in self.arms.items()}
        tmp = self.rec / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        os.replace(tmp, self.rec / "state.json")

    def _append(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        p = self.rec / name
        pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)

    # ---------------------------------------------------------------- model
    def ensure_model(self, now: pd.Timestamp, train_events=None) -> None:
        import lightgbm as lgb

        month = now.floor("D").replace(day=1)
        if self.model_month == month:
            return
        path = self.models_dir / f"{month:%Y-%m}.txt"
        if not path.exists():
            if train_events is None:
                raise FileNotFoundError(f"no model for {month:%Y-%m}: build it with train_events")
            ev = train_events(month)
            model = E.train(ev, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(model.model_to_string())
            log.info("trained the %s model on %d events", f"{month:%Y-%m}", int(np.isfinite(ev["label"]).sum()))
        self.model = lgb.Booster(model_str=path.read_text())
        self.model_month = month

    # ---------------------------------------------------------------- trading
    def _quote(self, book: dict, sym: str):
        b, a = book[sym]
        return b, a

    def due(self, now: pd.Timestamp) -> list[Position]:
        return [p for a in self.arms.values() for p in a.open if pd.Timestamp(p.exit_due) <= now]

    def exits(self, now: pd.Timestamp, book: dict, fund: dict | None = None) -> list[dict]:
        """Close every position whose holding time is up; funding = the settled rates in (entry, exit]."""
        out = []
        for arm in self.arms.values():
            keep = []
            for p in arm.open:
                if pd.Timestamp(p.exit_due) <= now:
                    bid, _ = self._quote(book, p.symbol)
                    px = bid * (1 - IMPACT_BPS * 1e-4)
                    r = (fund or {}).get(p.symbol)
                    funding = float(r[(r.index > pd.Timestamp(p.entry_time)) & (r.index <= now)].sum()) if r is not None and len(r) else 0.0
                    net = px / p.entry_px - 1.0 - 2 * TAKER_BPS * 1e-4 - funding
                    pnl = p.notional * net
                    arm.equity += pnl
                    out.append({"arm": arm.name, "symbol": p.symbol, "decided": p.decided, "entry_time": p.entry_time,
                                "entry_px": p.entry_px, "exit_time": str(now), "exit_px": px, "notional": p.notional,
                                "pred": p.pred, "gross": px / p.entry_px - 1.0, "fees": 2 * TAKER_BPS * 1e-4,
                                "funding": funding, "net": net, "pnl": pnl, "equity_after": arm.equity,
                                "late_s": (now - pd.Timestamp(p.exit_due)).total_seconds()})
                else:
                    keep.append(p)
            arm.open = keep
        self._append("trades.csv", out)
        return out

    def entries(self, decided: pd.Timestamp, ev: pd.DataFrame, pred: np.ndarray, now: pd.Timestamp, book: dict) -> list[dict]:
        rows = []
        order = np.argsort(-pred)
        for arm in self.arms.values():
            for k in order:
                if pred[k] <= arm.theta:
                    continue
                sym = ev["symbol"].iloc[k]
                if any(p.symbol == sym for p in arm.open):
                    continue
                nt = arm.size * arm.equity
                if sum(p.notional for p in arm.open) + nt > arm.max_gross * arm.equity:
                    continue
                _, ask = self._quote(book, sym)
                px = ask * (1 + IMPACT_BPS * 1e-4)
                arm.open.append(Position(arm.name, sym, str(decided), str(now), px, nt,
                                         str(now + pd.Timedelta(minutes=arm.hold)), float(pred[k])))
                rows.append({"arm": arm.name, "symbol": sym, "decided": str(decided), "entry_time": str(now),
                             "entry_px": px, "notional": nt, "pred": float(pred[k])})
        self._append("entries.csv", rows)
        return rows

    def mark(self, now: pd.Timestamp, last_close: dict[str, float]) -> list[dict]:
        rows = []
        for arm in self.arms.values():
            unreal = sum(p.notional * (last_close.get(p.symbol, p.entry_px) / p.entry_px - 1.0) for p in arm.open)
            rows.append({"t": str(now), "arm": arm.name, "equity": arm.equity + unreal, "realised": arm.equity,
                         "open": len(arm.open), "gross": sum(p.notional for p in arm.open)})
        self._append("equity.csv", rows)
        return rows

    def decide(self, T: pd.Timestamp, m: Minutes, met: dict, fund: dict, now: pd.Timestamp, book: dict) -> pd.DataFrame:
        """Score the 5-minute close T (decision at T + 1 minute) and open the arms' trades."""
        ev = E.event_table(m, met, fund, start=T, end=T + FIVE)
        if len(ev):
            pred = E.predict(self.model, ev)
            ev = ev.assign(pred=pred)
            self.entries(T, ev, pred, now, book)
            self._append("decisions.csv", ev.assign(decided_at=str(now))[["t", "symbol", "z", "doi_z", "pred", "decided_at"]]
                         .to_dict("records"))
        return ev


def settlements(a: pd.Timestamp, b: pd.Timestamp) -> list[pd.Timestamp]:
    """Funding settlements (00/08/16 UTC) in (a, b]."""
    days = pd.date_range(a.floor("D"), b.floor("D"), freq="D")
    return [d + pd.Timedelta(hours=h) for d in days for h in (0, 8, 16) if a < d + pd.Timedelta(hours=h) <= b]


def run_shift(rec: Path, cfg, minutes: float, end_at: pd.Timestamp, cache: Path | None = None, api=None,
              clock=None, sleep=time.sleep, train_events=None, step: float = 5.0, buffer_days: int = BUFFER_DAYS,
              model=None, symbols: list[str] | None = None, arms: list[dict] | None = None,
              half_spread: dict | None = None) -> dict:
    """One shift: load the buffer, then every ``step`` seconds trade due exits and new 5-minute closes."""
    api = api or BinanceLive()
    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    syms = symbols or cfg.intraday.symbols
    eng = LiveEngine(rec, syms, api, arms=arms, half_spread=half_spread)
    t0 = clock()
    eng.state.setdefault("started", str(t0))
    stop = min(t0 + pd.Timedelta(minutes=minutes), end_at)
    buf = Buffer(api, syms, cache, buffer_days)
    buf.load(t0)
    if model is not None:
        eng.model, eng.model_month = model, t0.floor("D").replace(day=1)
    else:
        eng.ensure_model(t0, train_events)
    last_T = pd.Timestamp(eng.state["last_decision"]) if eng.state.get("last_decision") else None
    last_mark = pd.Timestamp(eng.state["last_mark"]) if eng.state.get("last_mark") else t0.floor("5min")
    missed = failures = 0
    while True:
        now = clock()
        if now >= stop:
            break
        try:
            buf.update(now)
            book = api.book()
            if model is None:
                eng.ensure_model(now, train_events)
            closing = eng.due(now)
            if closing and any(settlements(pd.Timestamp(p.entry_time), now) and
                               (not len(buf.fund.get(p.symbol, [])) or buf.fund[p.symbol].index[-1] < settlements(pd.Timestamp(p.entry_time), now)[-1])
                               for p in closing):
                buf.refresh_funding(now)                         # a settlement inside a closing position's window
            eng.exits(now, book, buf.fund)
            T = (now - MIN).floor("5min")                       # the newest 5-minute close whose minute has closed
            if last_T is None or T > last_T:
                if now - (T + MIN) <= pd.Timedelta(minutes=2):  # fresh enough to trade
                    m = buf.minutes()
                    need = T - E.METRICS_LAG * FIVE                 # the newest metrics row the decision uses
                    late = now - (T + MIN) >= DECIDE_WAIT
                    if m.index[-1] >= T and (late or all(buf.complete(s, need) for s in syms)):
                        if late:
                            log.warning("deciding %s without complete metrics for %s", T,
                                        [s for s in syms if not buf.complete(s, need)])
                        eng.decide(T, m, buf.met, buf.fund, now, book)
                        last_T = T
                else:
                    missed += 1
                    log.warning("skipped the 5-minute close %s (now %s): too late to trade", T, now)
                    last_T = T
            if now.floor("5min") > last_mark:
                closes = {s: float(buf.bars[s]["close"].iloc[-1]) for s in syms if len(buf.bars[s])}
                eng.mark(now, closes)
                last_mark = now.floor("5min")
            eng.state.update({"last_decision": str(last_T) if last_T is not None else None, "last_mark": str(last_mark),
                              "updated": str(now)})
            eng.save()
            failures = 0
        except Exception as exc:  # noqa: BLE001 - one bad cycle (network, odd reply) must not end the shift
            failures += 1
            log.warning("cycle failed (%d in a row): %s", failures, exc, exc_info=not isinstance(exc, ConnectionError))
            if failures >= MAX_FAILURES:
                raise
        sleep(step)
    buf.save()
    eng.state["missed_closes"] = int(eng.state.get("missed_closes", 0)) + missed
    eng.save()
    return eng.state


def liquidate(rec: Path, cfg, api=None, now=None) -> dict:
    """Close every open position at the live bid (the experiment's end)."""
    api = api or BinanceLive()
    now = now or pd.Timestamp.now(tz="UTC")
    eng = LiveEngine(rec, cfg.intraday.symbols, api)
    for arm in eng.arms.values():
        for p in arm.open:
            p.exit_due = str(min(pd.Timestamp(p.exit_due), now))
    fund = {s: api.funding(s, now - pd.Timedelta(days=2)) for s in {p.symbol for a in eng.arms.values() for p in a.open}}
    eng.exits(now, api.book(), fund)
    eng.save()
    return {n: a.equity for n, a in eng.arms.items()}


# ---------------------------------------------------------------------------------------- the experiment
ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DAYS = 28
BRANCH = "experiment/daytrade-live"


def summarise(rec: Path) -> dict:
    """status.json: the state of each arm (realised and marked equity, trades, results)."""
    rec = Path(rec)
    st = json.loads((rec / "state.json").read_text()) if (rec / "state.json").exists() else {}
    tr = pd.read_csv(rec / "trades.csv") if (rec / "trades.csv").exists() else pd.DataFrame()
    eq = pd.read_csv(rec / "equity.csv") if (rec / "equity.csv").exists() else pd.DataFrame()
    out = {"experiment": "daytrade-live-4w", "started": st.get("started"), "ends": st.get("ends"),
           "updated": st.get("updated"), "missed_5min_closes": st.get("missed_closes", 0), "arms": {}}
    for name, a in st.get("arms", {}).items():
        t = tr[tr["arm"] == name] if len(tr) else tr
        last = eq[eq["arm"] == name].iloc[-1] if len(eq) and (eq["arm"] == name).any() else None
        marked = float(last["equity"]) if last is not None else a["equity"]
        out["arms"][name] = {"realised_equity": round(a["equity"], 2), "marked_equity": round(marked, 2),
                             "return": round(marked / 100_000 - 1, 6), "closed_trades": int(len(t)),
                             "open_positions": len(a["open"]),
                             "win_rate": round(float((t["net"] > 0).mean()), 4) if len(t) else None,
                             "net_bp_per_trade": round(float(1e4 * t["net"].mean()), 2) if len(t) else None,
                             "fees_and_funding_bp": round(float(1e4 * (t["fees"] + t["funding"]).mean()), 2) if len(t) else None}
    (rec / "status.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def archive_events(cfg, month: pd.Timestamp) -> pd.DataFrame:
    """Every event before ``month`` from the public archive (to train a month's model when none is committed)."""
    from tradinglab.daytrade import data as D
    from tradinglab.data.store import MarketData

    md = MarketData(cfg.paths.data_dir)
    syms = cfg.intraday.symbols
    m = D.minutes(md, syms, "2022-07-01", month + pd.Timedelta(hours=1))
    met = {s: D.metrics(md.archive, s, "2022-07-01", month) for s in syms}
    fund = {s: md.funding_8h(s, pd.Timestamp("2022-06-01")) for s in syms}
    return E.event_table(m, met, fund, end=month)


def experiment_run(cfg, rec: Path, minutes: float = 330.0, cache: Path | None = None, commit_every: float = 60.0,
                   branch: str = BRANCH) -> dict:
    """One shift of the 4-week live experiment; commits the records every ``commit_every`` minutes."""
    from tradinglab.experiment import git_push

    rec = Path(rec)
    rec.mkdir(parents=True, exist_ok=True)
    if (rec / "FINISHED.json").exists():
        log.info("the experiment has finished")
        return json.loads((rec / "FINISHED.json").read_text())
    (rec / "models").mkdir(exist_ok=True)
    for f in (ROOT / "models" / "daytrade2").glob("*.txt"):          # committed monthly models
        if not (rec / "models" / f.name).exists():
            (rec / "models" / f.name).write_text(f.read_text())
    st = json.loads((rec / "state.json").read_text()) if (rec / "state.json").exists() else {}
    started = pd.Timestamp(st["started"]) if st.get("started") else pd.Timestamp.now(tz="UTC")
    end_at = started + pd.Timedelta(days=EXPERIMENT_DAYS)
    if not st:
        st = {"started": str(started), "ends": str(end_at), "arms_spec": "config/daytrade_forward_arms.json",
              "candidate_spec": "config/daytrade2_candidate.json", "metrics_lag": E.METRICS_LAG}
        (rec / "state.json").write_text(json.dumps(st, indent=1))
    shift_end = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=minutes)
    git_push(rec, branch, f"daytrade-live: shift started {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC")
    while True:
        now = pd.Timestamp.now(tz="UTC")
        left = min((shift_end - now).total_seconds() / 60, (end_at - now).total_seconds() / 60)
        if left <= 0.5:
            break
        run_shift(rec, cfg, min(commit_every, left), end_at, cache=cache, train_events=lambda m: archive_events(cfg, m))
        s = summarise(rec)
        git_push(rec, branch, "daytrade-live: " + ", ".join(f"{n} {a['return']:+.2%} ({a['closed_trades']} trades)"
                                                          for n, a in s["arms"].items()))
    if pd.Timestamp.now(tz="UTC") >= end_at:
        eq = liquidate(rec, cfg)
        s = summarise(rec)
        fin = {"finished": str(pd.Timestamp.now(tz="UTC")), "started": str(started), "final_equity": eq, "status": s}
        (rec / "FINISHED.json").write_text(json.dumps(fin, indent=1, default=str) + "\n")
        git_push(rec, branch, "daytrade-live: finished – " + ", ".join(f"{n} {v:,.2f}" for n, v in eq.items()))
        return fin
    return summarise(rec)


def experiment_stop(cfg, rec: Path, reason: str = "stopped early on request", branch: str = BRANCH) -> dict:
    """End the live experiment now: close any open position at the live bid and write FINISHED.json."""
    from tradinglab.experiment import git_push

    rec = Path(rec)
    if (rec / "FINISHED.json").exists():
        return json.loads((rec / "FINISHED.json").read_text())
    st = json.loads((rec / "state.json").read_text())
    eq = liquidate(rec, cfg)
    s = summarise(rec)
    fin = {"finished": str(pd.Timestamp.now(tz="UTC")), "started": st.get("started"), "planned_end": st.get("ends"),
           "stopped_early": True, "reason": reason, "final_equity": eq, "status": s}
    (rec / "FINISHED.json").write_text(json.dumps(fin, indent=1, default=str) + "\n")
    git_push(rec, branch, "daytrade-live: stopped early – " + ", ".join(f"{n} {v:,.2f}" for n, v in eq.items()))
    return fin
