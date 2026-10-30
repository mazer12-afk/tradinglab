"""Fast-forward simulation: the live engines on archived market data, minutes instead of weeks.

A simulation runs the *same* engines, accounts, trading rules and profit-vault twins as a live experiment
(``tradinglab.experiment``), but over a past window of the public Binance archive and driven by a simulated
clock, so two weeks of trading take minutes. Only three things differ from live trading:

* **Prices.** The archive has bars, not order books. A fill crosses a modelled spread around the archived
  price: the 5-minute bar close for the intraday book, the open of the current hour for the daily book (the
  1-minute open for cycles inside an hour, such as the start and the close-out), ± each coin's median live
  half-spread (``config/sim_spreads.json``, measured from the vault-2w live fills).
* **Funding.** Settled funding rates are charged: OKX history for the intraday book (its live source; OKX
  keeps about three months, older settlements use the Binance archive), the Binance archive for the daily
  book, 1 bp per 8h where none is known. The daily models see only the funding a live run could see: the
  Binance archive publishes it monthly, so the current month is unknown to them, as it is live.
* **Speed.** Model signals are computed once for the whole window instead of every bar, the intraday book is
  sized on the last two weeks of data instead of the 35-day buffer (the EWMA sizing span is one day, so the
  difference is below 1e-12), and the daily book cycles hourly instead of every five minutes.

Out-of-sample guard: a model must not be simulated on data it was trained on. ``simulate`` refuses windows
that start before the models' training data ended (``models/intraday/live/manifest.json`` and
``models/selection.json``) unless ``allow_in_sample`` is set, which marks the results as in-sample.

Walk-forward (``retrain``): for windows in the past, the models are retrained at the start and on the first
of every month, each time only on data closed by then (the lab's monthly retraining), and the engines
restart on the new models as a live deployment would. LightGBM trains in about a minute per step; the deep
models (about 30 minutes each per step) are retrained only with ``retrain_deep`` and are otherwise left out.
The production blend ("main") was chosen by the research tournament on data up to ``selection.json``'s
``data_end``; before that date it is left out of walk-forward runs unless ``allow_in_sample`` is set.
"""

from __future__ import annotations

import copy
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tradinglab.config import LabConfig, utc
from tradinglab.data.exchanges import Quote
from tradinglab.data.store import MarketData, funding_schedule
from tradinglab.intraday import live as IL
from tradinglab.intraday.data import IntradayData, assemble_intraday
from tradinglab.intraday.features import build_features
from tradinglab.intraday.policy import PolicySpec, policies_by_name
from tradinglab.live.engine import LiveEngine, load_selection
from tradinglab.live.feed import LiveFeed
from tradinglab.live.state import StateStore

log = logging.getLogger(__name__)

BAR = pd.Timedelta(minutes=5)
SIZING_BARS = 288 * 14          # sizing window (EWMA span 288 → weight of older data < 1e-12)
QUOTE_LATENCY = pd.Timedelta(seconds=8)   # live quotes were taken ~8 s after the bar close
SPREADS_FILE = Path(__file__).resolve().parents[2] / "config" / "sim_spreads.json"


class InSampleError(ValueError):
    pass


@dataclass
class SimClock:
    now: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("1970-01-01", tz="UTC"))


def load_spreads(path: Path | None = None) -> dict[str, float]:
    """Per-coin half-spread as a fraction of the mid (default 1 bp for coins not listed)."""
    p = Path(path) if path else SPREADS_FILE
    return json.loads(p.read_text())["half_spread"] if p.exists() else {}


def quote(mid: float, symbol: str, t: pd.Timestamp, spreads: dict[str, float]) -> Quote:
    h = spreads.get(symbol, 1e-4)
    return Quote(mid * (1 - h), mid * (1 + h), t, "sim")


def training_end(cfg: LabConfig) -> dict[str, pd.Timestamp | None]:
    """When the models' training data ended (simulating before that is in-sample)."""
    out: dict[str, pd.Timestamp | None] = {"intraday": None, "daily": None}
    man = Path(cfg.paths.models_dir) / "intraday" / "live" / "manifest.json"
    if man.exists():
        out["intraday"] = utc(json.loads(man.read_text())["data_end"])
    sel = load_selection(cfg)
    if sel.get("data_end"):
        out["daily"] = utc(sel["data_end"])
    return out


def check_out_of_sample(cfg: LabConfig, start: pd.Timestamp, books: list[str], allow: bool,
                        retrain: bool = False) -> list[str]:
    notes = []
    for book, end in training_end(cfg).items():
        if book not in books or end is None or start >= end:
            continue
        if retrain and book == "intraday":
            continue                    # every model is trained on data before its walk-forward step
        if retrain:
            notes.append(f"daily: the production blend ('main') was selected on data up to {end}; "
                         + ("kept (in-sample)" if allow else "left out"))
            continue
        msg = (f"{book} models were trained on data up to {end}; a simulation from {start} is in-sample "
               f"and flatters them")
        if not allow:
            raise InSampleError(msg + " (use walk-forward retraining, or allow_in_sample to proceed)")
        log.warning(msg)
        notes.append(msg)
    return notes


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Walk-forward steps: ``start``, then the first of every month before ``end``."""
    steps, t = [start], start
    while True:
        t = pd.Timestamp(year=t.year + t.month // 12, month=t.month % 12 + 1, day=1, tz="UTC")
        if t >= end:
            return steps
        steps.append(t)


# ------------------------------------------------------------------ intraday book
def intraday_data(md: MarketData, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> IntradayData:
    """Archived 5-minute spot bars with close time in [start, end], assembled exactly as the live feed does."""
    kl = {}
    for s in symbols:
        k = md.klines(s, "5m", start.floor("D").tz_localize(None), market="spot")
        k = k.drop(columns=[c for c in ("ignore", "close_time") if c in k])
        kl[s] = k[(k.index + BAR >= start) & (k.index + BAR <= end)]
    return assemble_intraday(kl, {}, {}, "5m", 0.0)


class SimIntradayFeed(IL.IntradayFeed):
    """The price buffer, served from a precomputed archive frame up to the simulated clock."""

    def __init__(self, data: IntradayData, clock: SimClock, quotes_fn, window_bars: int, first_window_bars: int):
        super().__init__(md=None, symbols=data.symbols, exchange=object())
        self.full, self.clock, self.quotes_fn = data, clock, quotes_fn
        self.window, self.first_window = window_bars, first_window_bars
        self.served = 0

    def load(self, now=None) -> None:
        pass

    def update(self) -> None:
        pass

    def last_common_close(self, with_lagging: bool = False, max_lag=None):
        idx = self.full.index
        t = idx[idx.searchsorted(self.clock.now, side="right") - 1]
        return (t, []) if with_lagging else t

    def data(self, until=None) -> IntradayData:
        until = until or self.last_common_close()
        i = self.full.index.searchsorted(until, side="right")
        n = self.first_window if self.served == 0 else self.window   # the first cycle warms the scorers up
        self.served += 1
        return self.full.slice(self.full.index[max(0, i - n)], until)

    def quotes(self):
        return self.quotes_fn()


class SimIntradayEngine(IL.IntradayEngine):
    """The live intraday engine with precomputed model signals and archived funding."""

    def __init__(self, *a, signals: dict[str, pd.DataFrame], funding: dict[str, pd.Series], **kw):
        super().__init__(*a, **kw)
        self.signals = signals
        self.funding_hist = funding

    def _features(self, d):
        return None

    def _outputs(self, spec: PolicySpec, fs, d: IntradayData, rows: np.ndarray) -> np.ndarray:
        return self.signals[spec.signal_key].reindex(d.index[rows]).to_numpy()

    def _refresh_funding(self, now) -> None:
        pass

    def _funding_rate(self, sym: str, stamp: pd.Timestamp) -> tuple[float, bool]:
        r = self.funding_hist.get(sym)
        if r is not None and stamp in r.index:
            return float(r[stamp]), False
        return 0.0001, True

    def _export(self, rep, close=None) -> None:
        pass


def precompute_signals(cfg: LabConfig, models_dir: Path, data: IntradayData, specs: list[PolicySpec],
                       first: pd.Timestamp, last: pd.Timestamp | None = None, fs=None) -> dict[str, pd.DataFrame]:
    """Every strategy's model/rule output for every bar in [first, last], through the engine's own code."""
    fs = build_features(data, 12) if fs is None else fs
    rows = np.flatnonzero((data.index >= first) & (data.index <= (last if last is not None else data.index[-1])))
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        store = StateStore(Path(tmp) / "signals.sqlite")
        base = IL.IntradayEngine(cfg, SimpleNamespace(symbols=data.symbols), store, models_dir, policies=specs,
                                 vault_thresholds=[])
        for spec in specs:
            if spec.signal_key in out:
                continue
            t0 = time.time()
            vals = base._outputs(spec, fs, data, rows)
            out[spec.signal_key] = pd.DataFrame(vals, index=data.index[rows], columns=data.symbols)
            log.info("signals %s: %d bars in %.1fs", spec.signal_key, len(rows), time.time() - t0)
        store.close()
    return out


def okx_funding_history(symbols: list[str], start: pd.Timestamp) -> dict[str, pd.Series]:
    """Realised OKX USDT-swap funding per settlement since ``start`` (the live intraday engine's source)."""
    import requests

    out = {}
    for s in symbols:
        rows, after = [], ""
        try:
            for _ in range(200):
                r = requests.get(f"{IL.OKX}/api/v5/public/funding-rate-history",
                                 params={"instId": f"{s[:-4]}-USDT-SWAP", "limit": 100, **({"after": after} if after else {})},
                                 timeout=15).json().get("data", [])
                rows += r
                if not r or int(r[-1]["fundingTime"]) < start.timestamp() * 1000:
                    break
                after = r[-1]["fundingTime"]
        except Exception as exc:  # noqa: BLE001 - funding is small; fall back to the default rate
            log.warning("funding history %s unavailable: %s", s, exc)
        ser = pd.Series({pd.Timestamp(int(x["fundingTime"]), unit="ms", tz="UTC"):
                         float(x.get("realizedRate") or x["fundingRate"]) for x in rows}, dtype="float64")
        out[s] = ser.sort_index()
    return out


def intraday_funding(md: MarketData, symbols: list[str], start: pd.Timestamp) -> dict[str, pd.Series]:
    """Funding per settlement: OKX history where it reaches, the Binance archive before that."""
    okx = okx_funding_history(symbols, start)
    out = {}
    for s in symbols:
        bn = md.funding_8h(s, start.floor("D").tz_localize(None))
        if len(bn):
            bn = bn.copy()
            bn.index = bn.index.round("min")
        out[s] = okx.get(s, pd.Series(dtype="float64")).combine_first(bn).sort_index()
    return out


def train_intraday_step(cfg: LabConfig, data: IntradayData, specs: list[PolicySpec], step: pd.Timestamp,
                        out: Path, kinds: tuple[str, ...]) -> Path:
    """Retrain the intraday models on the bars closed by ``step`` (the lab's monthly retraining)."""
    from tradinglab.intraday.train import train_live_intraday

    mdir = out / "models" / "intraday" / f"{step:%Y%m%d}"
    t0 = time.time()
    train_live_intraday(cfg, specs, out_dir=mdir, kinds=kinds, history=data.slice(data.index[0], step))
    log.info("walk-forward: intraday models trained on data to %s in %.0fs", step, time.time() - t0)
    return mdir


def run_intraday(cfg: LabConfig, start: pd.Timestamp, end: pd.Timestamp, out: Path,
                 policies: list[PolicySpec] | None = None, thresholds: list[float] | None = None,
                 spreads: dict[str, float] | None = None, md: MarketData | None = None,
                 funding: dict[str, pd.Series] | None = None, models_dir: Path | None = None,
                 data: IntradayData | None = None, latency: pd.Timedelta = QUOTE_LATENCY,
                 retrain: bool = False, retrain_deep: bool = False) -> dict:
    """Simulate the intraday book over (start, end] into ``out/intraday.sqlite``."""
    md = md or MarketData(cfg.paths.data_dir)
    specs = policies or policies_by_name(cfg.intraday.live_policies)
    dropped = [p.name for p in specs if retrain and not retrain_deep and p.model == "deep"]
    specs = [p for p in specs if p.name not in dropped]
    kinds = ("lgbm", "deep") if retrain_deep else ("lgbm",)
    spreads = load_spreads() if spreads is None else spreads
    warm = max((s.norm_window + s.horizon for s in specs if s.rule == "selective"), default=0) + 12
    first_window = max(warm, SIZING_BARS) + 1
    if data is None:
        lookback = (pd.Timedelta(days=cfg.intraday.train_days + 45) if retrain
                    else BAR * (first_window + 288 * 8))
        data = intraday_data(md, cfg.intraday.symbols, start - lookback, end)
    if funding is None:
        funding = intraday_funding(md, cfg.intraday.symbols, start - pd.Timedelta(days=1))
    t0 = time.time()
    fs = build_features(data, 12)
    steps = month_starts(start, end) if retrain else [start]

    def signals_at(k: int) -> tuple[Path, dict]:
        step = steps[k]
        mdir = (train_intraday_step(cfg, data, specs, step, out, kinds) if retrain
                else Path(models_dir or Path(cfg.paths.models_dir) / "intraday" / "live"))
        first = data.index[max(0, data.index.searchsorted(step) - warm - 2)]
        last = steps[k + 1] if k + 1 < len(steps) else end
        return mdir, precompute_signals(cfg, mdir, data, specs, first, last, fs=fs)

    clock = SimClock()
    close = data.close

    def quotes_fn():
        t = clock.now
        i = close.index.searchsorted(t, side="right") - 1
        row = close.iloc[i]
        return {s: quote(float(row[s]), s, t, spreads) for s in data.symbols if np.isfinite(row[s])}

    feed = SimIntradayFeed(data, clock, quotes_fn, SIZING_BARS, first_window)
    store = StateStore(out / "intraday.sqlite")
    mdir, signals = signals_at(0)
    eng = SimIntradayEngine(cfg, feed, store, mdir, policies=specs, vault_thresholds=thresholds,
                            signals=signals, funding=funding)
    bars = data.index[(data.index >= start.floor("5min")) & (data.index <= end)]
    nxt = 1
    for k, t in enumerate(bars):
        clock.now = max(t + latency, start) if k == 0 else t + latency
        if nxt < len(steps) and clock.now >= steps[nxt]:
            # retrained models: restart as a deployment would (scores warm up again on the new forecasts)
            eng.models_dir, eng.signals = signals_at(nxt)
            eng.manifest = json.loads((eng.models_dir / "manifest.json").read_text())
            eng.scorers = {}
            feed.served = 0
            nxt += 1
        eng.cycle(clock.now)
        if k % 1000 == 0:
            log.info("intraday %s (%d/%d, %.0fs)", t, k, len(bars), time.time() - t0)
    clock.now = end
    eng.liquidate(end, reason="simulation end")
    store.close()
    return {"bars": len(bars), "seconds": round(time.time() - t0, 1), "steps": [str(x) for x in steps],
            "left_out": dropped}


# ------------------------------------------------------------------ daily book
class SimDailyFeed(LiveFeed):
    """Archived hourly bars up to the simulated clock; quotes at the open of the current hour ± spread
    (cycles more than a minute into the hour, such as the start and the close-out, use the archived
    1-minute open instead).

    Funding: the models see what the monthly Binance archive had published (settlements before this month);
    accounts are charged the settled rates.

    Prices, the data horizon and funding are read from arrays built once (the live feed rebuilds a wide frame
    since 2018 every cycle and its funding schedule on every call: fine every five minutes, the bulk of the
    time in a simulation). They return the same values as the live feed's methods.
    """

    def __init__(self, hourly: dict[str, pd.DataFrame], rates: dict[str, pd.Series], clock: SimClock,
                 spreads: dict[str, float], default_funding_8h: float,
                 span: tuple[pd.Timestamp, pd.Timestamp] | None = None, md: MarketData | None = None):
        super().__init__(md=md, symbols=list(hourly), history_start="2018-01-01", mode="archive", exchanges=[],
                         default_funding_8h=default_funding_8h)
        self.hourly_full, self.rates_full, self.clock, self.spreads = hourly, rates, clock, spreads
        self._ends = {s: h.index + pd.Timedelta(hours=1) for s, h in hourly.items()}
        self._close = {s: h["close"].to_numpy(dtype="float64") for s, h in hourly.items()}
        self._cut: dict[str, int] = {}
        self.span = span
        self._sched: dict[str, tuple[pd.Series, pd.Series]] = {}

    def refresh(self) -> None:
        self.mode = "realtime"
        now = self.clock.now
        self._cut = {s: int(self._ends[s].searchsorted(now, side="right")) for s in self.hourly_full}
        self.hourly = {s: h.iloc[: self._cut[s]] for s, h in self.hourly_full.items()}
        published = pd.Timestamp(year=now.year, month=now.month, day=1, tz="UTC")
        self.rates = {s: r[r.index < published] if len(r) else r for s, r in self.rates_full.items()}
        self._closes = None

    def horizon(self, symbols: list[str] | None = None, active_days: int = 7) -> pd.Timestamp:
        last = {s: self._ends[s][n - 1] for s, n in self._cut.items() if n > 0 and (symbols is None or s in symbols)}
        if not last:
            raise RuntimeError("no market data")
        newest = max(last.values())
        return min(t for t in last.values() if t >= newest - pd.Timedelta(days=active_days))

    def price_at(self, symbol: str, ts: pd.Timestamp) -> float | None:
        if symbol not in self._cut:
            return None
        k = min(int(self._ends[symbol].searchsorted(ts, side="right")), self._cut[symbol]) - 1
        return float(self._close[symbol][k]) if k >= 0 else None

    def funding_events(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple]:
        if end <= start:
            return []
        rates = self.rates_full.get(symbol, pd.Series(dtype="float64"))
        if self.span is not None and start >= self.span[0] - pd.Timedelta(days=1) and end <= self.span[1]:
            if symbol not in self._sched:
                self._sched[symbol] = funding_schedule(rates, self.span[0] - pd.Timedelta(days=2), self.span[1],
                                                       self.default_funding_8h)
            sched, est = self._sched[symbol]
        else:
            sched, est = funding_schedule(rates, start - pd.Timedelta(days=1), end, self.default_funding_8h)
        i, j = sched.index.searchsorted(start, side="right"), sched.index.searchsorted(end, side="right")
        if i >= j:
            return []
        return [(t, float(r), bool(e)) for t, r, e in zip(sched.index[i:j], sched.iloc[i:j], est.iloc[i:j])]

    def minute_open(self, symbol: str, t: pd.Timestamp) -> float | None:
        """Open of the archived 1-minute bar containing ``t`` (None if unavailable)."""
        day = t.floor("D")
        key = (symbol, str(day.date()))
        if key not in self._minute_cache:
            try:
                self._minute_cache[key] = self.md.archive.klines(symbol, "1m", day, day + pd.Timedelta(days=1))
            except Exception as exc:  # noqa: BLE001 - fall back to the hourly open
                log.debug("1m klines %s %s unavailable: %s", symbol, day.date(), exc)
                self._minute_cache[key] = pd.DataFrame()
        df = self._minute_cache[key]
        minute = t.floor("min")
        if df.empty or minute not in df.index:
            return None
        return float(df.loc[minute, "open"])

    def quote(self, symbol: str) -> Quote | None:
        now = self.clock.now
        if self.md is not None and now - now.floor("h") >= pd.Timedelta(minutes=1):
            px = self.minute_open(symbol, now)
            if px is not None:
                return quote(px, symbol, now, self.spreads)
        h = self.hourly_full.get(symbol)
        if h is None or h.empty:
            return None
        hour = self.clock.now.floor("h")
        i = h.index.searchsorted(hour, side="right") - 1
        if i < 0 or h.index[i] != hour:
            return None
        return quote(float(h["open"].iloc[i]), symbol, self.clock.now, self.spreads)


class SimLiveEngine(LiveEngine):
    """The live daily engine; ``drop`` leaves strategies (with their vault twins) out of the roster."""

    def __init__(self, *a, drop: tuple[str, ...] = (), **kw):
        self.drop = set(drop)
        super().__init__(*a, **kw)

    def _account_specs(self):
        return [s for s in super()._account_specs() if s.name.split("+vault")[0] not in self.drop]

    def _export(self, rep) -> None:
        pass


def train_daily_step(cfg: LabConfig, md: MarketData, kinds: list[str], step: pd.Timestamp, out: Path) -> dict:
    """Refit the daily deep models on data before ``step`` → {name: path}."""
    from tradinglab.research.pipeline import train_live_models

    c = copy.deepcopy(cfg)
    c.paths.models_dir = str(out / "models" / "daily" / f"{step:%Y%m%d}")
    panel = md.panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start, end=step,
                     exec_lag_hours=cfg.market.exec_lag_hours, default_funding_8h=cfg.market.default_funding_8h)
    return train_live_models(c, panel, kinds)


def run_daily(cfg: LabConfig, start: pd.Timestamp, end: pd.Timestamp, out: Path,
              spreads: dict[str, float] | None = None, md: MarketData | None = None,
              selection: dict | None = None, retrain: bool = False, retrain_deep: bool = False,
              drop_main: bool = False) -> dict:
    """Simulate the daily book over (start, end] into ``out/lab.sqlite`` (hourly cycles)."""
    md = md or MarketData(cfg.paths.data_dir)
    sel = copy.deepcopy(selection if selection is not None else load_selection(cfg))
    kinds = [n[3:] for n in sel.get("deep_models", {})]
    dropped = (["main"] if drop_main else []) + ([f"shadow:dl_{k}" for k in kinds] if retrain and not retrain_deep
                                                 else [])
    if retrain and not retrain_deep:
        sel["deep_models"] = {}          # untrained deep strategies get no account
    spreads = load_spreads() if spreads is None else spreads
    syms = cfg.universe.symbols
    hist = pd.Timestamp(cfg.walkforward.data_start)
    hourly = {s: md.hourly(s, hist) for s in syms}
    hourly = {s: h[h.index < end] for s, h in hourly.items()}   # the open of the current hour is known
    rates = {s: md.funding_8h(s, hist) for s in syms}
    t0 = time.time()
    clock = SimClock()
    feed = SimDailyFeed(hourly, rates, clock, spreads, cfg.market.default_funding_8h, span=(start, end), md=md)
    store = StateStore(out / "lab.sqlite")
    steps = month_starts(start, end) if retrain else [start]

    def engine_at(step: pd.Timestamp) -> SimLiveEngine:
        if retrain and retrain_deep and kinds:
            sel["deep_models"] = train_daily_step(cfg, md, kinds, step, out)
        return SimLiveEngine(cfg, feed, store, sel, drop=("main",) if drop_main else ())

    eng = engine_at(start)
    times = [start] + list(pd.date_range(start.floor("h") + pd.Timedelta(hours=1, seconds=30), end, freq="h"))
    nxt = 1
    for k, t in enumerate(times):
        clock.now = t
        if nxt < len(steps) and t >= steps[nxt]:
            eng = engine_at(steps[nxt])      # restart on the retrained models
            nxt += 1
        eng.cycle(t)
        if k % 100 == 0:
            log.info("daily %s (%d/%d, %.0fs)", t, k, len(times), time.time() - t0)
    clock.now = end
    eng.cycle(end)
    eng.liquidate(end, reason="simulation end")
    store.close()
    return {"cycles": len(times), "seconds": round(time.time() - t0, 1), "steps": [str(x) for x in steps],
            "left_out": dropped}


# ------------------------------------------------------------------ results
def summarise(out: Path) -> pd.DataFrame:
    """Per account: equity at the end, vault, total, return, sweeps, fills and costs."""
    rows = []
    for db in ("intraday.sqlite", "lab.sqlite"):
        if not (out / db).exists():
            continue
        st = StateStore(out / db)
        for a in st.accounts():
            name = a["name"]
            last = st.last_equity(name)
            eq = last["equity"] if last else a["initial_capital"]
            vault = st.vault_total(name)
            o = st.orders(name, limit=10_000_000)
            o = o[o["status"] == "filled"] if len(o) else o
            rows.append({"book": "intraday" if db.startswith("intraday") else "daily", "account": name,
                         "equity": eq, "vault": vault, "total": eq + vault,
                         "return": (eq + vault) / a["initial_capital"] - 1.0,
                         "sweeps": int(len(st.vault_events(name))), "fills": int(len(o)),
                         "fees": float(o["fee"].sum()) if len(o) else 0.0,
                         "spread": float(o["slippage_cost"].sum()) if len(o) else 0.0,
                         "funding": float(st.funding_total(name))})
        st.close()
    return pd.DataFrame(rows)


def simulate(cfg: LabConfig, start, end, out: Path, books: tuple[str, ...] = ("intraday", "daily"),
             allow_in_sample: bool = False, spreads_path: Path | None = None, retrain: bool = False,
             retrain_deep: bool = False) -> dict:
    """Run a simulation and write ``summary.csv`` and ``simulation.json`` next to the state files."""
    start, end = utc(start), utc(end)
    if end <= start:
        raise ValueError("end must be after start")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f).exists() for f in ("intraday.sqlite", "lab.sqlite")):
        raise FileExistsError(f"{out} already holds a simulation; choose an empty directory")
    notes = check_out_of_sample(cfg, start, list(books), allow_in_sample, retrain)
    sel_end = training_end(cfg)["daily"]
    drop_main = retrain and not allow_in_sample and sel_end is not None and start < sel_end
    cfg.paths.state_dir = str(out)
    spreads = load_spreads(spreads_path)
    md = MarketData(cfg.paths.data_dir)
    meta = {"start": str(start), "end": str(end), "books": list(books), "walk_forward": retrain,
            "retrain_deep": retrain_deep, "in_sample_notes": notes,
            "spreads": str(spreads_path or SPREADS_FILE), "runs": {}}
    if "intraday" in books:
        meta["runs"]["intraday"] = run_intraday(cfg, start, end, out, spreads=spreads, md=md, retrain=retrain,
                                                retrain_deep=retrain_deep)
    if "daily" in books:
        meta["runs"]["daily"] = run_daily(cfg, start, end, out, spreads=spreads, md=md, retrain=retrain,
                                          retrain_deep=retrain_deep, drop_main=drop_main)
    summary = summarise(out)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "simulation.json").write_text(json.dumps(meta, indent=1, default=str))
    return {"meta": meta, "summary": summary}
