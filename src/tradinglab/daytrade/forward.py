"""Forward paper test of the frozen day-trading candidate (config/daytrade2_candidate.json) on new days.

Binance's futures API is geo-blocked from the lab's servers, so the test trades the public archive instead:
each UTC day's 1-minute bars and futures metrics appear the next day. Day D is traded once D+1 is published
(a position opened late on D closes up to 4 hours into D+1). Nothing is re-tuned: rule, features, threshold
and the monthly retraining are exactly those committed before the test year, and every day after the frozen
test year (from 29 Oct 2026) counts.

Records (a directory, kept on its own git branch):
  events.parquet   every event with its features, label and forecast (seeded with history before the start)
  trades.csv       every simulated trade intent and its fill
  models/          the model used for each month
  equity.csv       daily account value; status.json  summary; state.json  last day traded
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.config import LabConfig
from tradinglab.daytrade import data as D
from tradinglab.daytrade import events as E
from tradinglab.daytrade.sim import Costs, book_stats, funding_cost, run_book, simulate_trades

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "config" / "daytrade2_candidate.json"
START = pd.Timestamp("2026-10-29", tz="UTC")       # first day after the frozen candidate's test year
WINDOW_DAYS = 21                                   # history loaded around each day (features need ~10)


def spec() -> dict:
    return json.loads(SPEC.read_text())["params"]


def load_state(rec: Path) -> dict:
    p = rec / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"start": str(START.date()), "last_day": None}


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")


def model_for(rec: Path, events: pd.DataFrame, month_start: pd.Timestamp):
    import lightgbm as lgb

    path = rec / "models" / f"{month_start:%Y-%m}.txt"
    if path.exists():
        return lgb.Booster(model_str=path.read_text())
    model = E.train(events, month_start)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_to_string())
    log.info("trained the model for %s on %d events", f"{month_start:%Y-%m}", int(np.isfinite(events["label"]).sum()))
    return model


def trade_day(rec: Path, cfg: LabConfig, md, day: pd.Timestamp) -> int:
    """Trade one UTC day; returns the number of trades taken into the records."""
    p = spec()
    syms = cfg.intraday.symbols
    lo, hi = day - pd.Timedelta(days=WINDOW_DAYS), day + pd.Timedelta(days=2)
    m = D.minutes(md, syms, lo, hi)
    met = {s: D.metrics(md.archive, s, lo, hi) for s in syms}
    fund = {s: md.funding_8h(s, (lo - pd.Timedelta(days=40)).tz_localize(None)) for s in syms}
    ev = E.event_table(m, met, fund, start=day, end=day + pd.Timedelta(days=1))
    store = pd.read_parquet(rec / "events.parquet")
    month = day.replace(day=1)
    model = model_for(rec, store, month)
    ev["pred"] = E.predict(model, ev) if len(ev) else []
    it = E.intents(ev, ev["pred"].to_numpy(), p["theta"], p["hold"], p["size"])
    costs = Costs(half_spread=json.loads((ROOT / "config" / "sim_spreads.json").read_text())["half_spread"])
    tr = simulate_trades(m, it, costs) if len(it) else it.assign(filled=False)
    if len(tr):
        tr["funding"] = funding_cost(tr, fund)
    store = pd.concat([store[store["t"] < day], ev], ignore_index=True)
    store.to_parquet(rec / "events.parquet")
    path = rec / "trades.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[pd.to_datetime(old["t"], utc=True) < day]
    pd.concat([old, tr], ignore_index=True).to_csv(path, index=False)
    log.info("%s: %d events, %d forecasts above %.0f bp, %d filled", day.date(), len(ev), len(it),
             1e4 * p["theta"], int(tr["filled"].sum()) if len(tr) else 0)
    return len(tr)


def summarise(rec: Path, through: pd.Timestamp) -> dict:
    p = spec()
    path = rec / "trades.csv"
    tr = pd.read_csv(path) if path.exists() else pd.DataFrame()
    status = {"strategy": "ml_event_filter_long", "spec": str(SPEC.relative_to(ROOT)), "start": str(START.date()),
              "through": str(through.date()), "days": int((through - START).days) + 1}
    if len(tr) and tr["filled"].any():
        for c in ("t", "entry_time", "exit_time"):
            tr[c] = pd.to_datetime(tr[c], utc=True)
        bk = run_book(tr, start=START, end=through + pd.Timedelta(days=1), max_gross=p["max_gross"])
        st = book_stats(bk)
        eq = (1 + bk.daily).cumprod() * 100
        eq.rename("equity_index").to_csv(rec / "equity.csv")
        status.update({k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for k, v in st.items()})
        status["total_return"] = round(float(eq.iloc[-1] / 100 - 1), 6) if len(eq) else 0.0
    else:
        status.update({"trades": 0, "total_return": 0.0})
    save_json(rec / "status.json", status)
    return status


def build_seed(cfg: LabConfig, md, first="2022-07-01") -> pd.DataFrame:
    """Every event before START with its features and label: the history the monthly models train on."""
    syms = cfg.intraday.symbols
    m = D.minutes(md, syms, first, START + pd.Timedelta(days=1))
    met = {s: D.metrics(md.archive, s, first, START + pd.Timedelta(days=1)) for s in syms}
    fund = {s: md.funding_8h(s, pd.Timestamp(first) - pd.Timedelta(days=30)) for s in syms}
    return E.event_table(m, met, fund, end=START)


def run(cfg: LabConfig, rec: Path, until=None, seed: Path | None = None) -> dict:
    """Trade every day that is published and not yet traded (building the history seed on the first run)."""
    from tradinglab.data.store import MarketData

    rec = Path(rec)
    rec.mkdir(parents=True, exist_ok=True)
    md = MarketData(cfg.paths.data_dir)
    if not (rec / "events.parquet").exists():
        s = pd.read_parquet(seed) if seed is not None else build_seed(cfg, md)
        s[s["t"] < START].to_parquet(rec / "events.parquet")
        log.info("history seed: %d events before %s", int((s["t"] < START).sum()), START.date())
    state = load_state(rec)
    last = pd.Timestamp(state["last_day"], tz="UTC") if state["last_day"] else START - pd.Timedelta(days=1)
    published = D.latest_complete_day(md.archive, cfg.intraday.symbols)
    final = published - pd.Timedelta(days=1)          # day D needs D+1 for exits and labels
    if until is not None:
        final = min(final, pd.Timestamp(until, tz="UTC").floor("D"))
    day = last + pd.Timedelta(days=1)
    while day <= final:
        trade_day(rec, cfg, md, day)
        state["last_day"] = str(day.date())
        save_json(rec / "state.json", state)
        day += pd.Timedelta(days=1)
    through = pd.Timestamp(state["last_day"], tz="UTC") if state["last_day"] else last
    return summarise(rec, through)
