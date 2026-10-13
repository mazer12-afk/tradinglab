"""Real-time intraday paper trading (5-minute bars).

Every cycle (the daemon polls every few seconds):

1. pull the bars that closed since the last cycle from the exchange REST API
   (history comes from the public archive);
2. once *every* symbol has the new bar, compute features on a rolling buffer,
   run each account's model and advance its trading policy;
3. mark every account at the bar close and charge perpetual funding at the
   00/08/16 UTC stamps using live funding rates;
4. send rebalancing orders that fill immediately at the live best bid/ask plus
   impact and taker fee.

Decisions only use bars that have closed; fills use quotes fetched *after*
the decision — there is no path for future information to leak in.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.config import CostConfig, LabConfig, bar_timedelta, utc
from tradinglab.data.exchanges import BinanceSpot, Quote
from tradinglab.data.store import MarketData
from tradinglab.intraday.data import IntradayData, assemble_intraday
from tradinglab.intraday.features import build_features
from tradinglab.intraday.policy import DEFAULT_POLICIES, PolicySpec, make_policy
from tradinglab.live.broker import PaperBroker
from tradinglab.live.state import StateStore
from tradinglab.portfolio.risk import DrawdownGuard

log = logging.getLogger(__name__)

OKX = "https://www.okx.com"


class IntradayFeed:
    """Rolling buffer of closed 5m bars: archive history + live exchange top-up."""

    def __init__(self, md: MarketData, symbols: list[str], bar: str = "5m", buffer_days: int = 35,
                 exchange: BinanceSpot | None = None):
        self.md = md
        self.symbols = list(symbols)
        self.bar = bar
        self.td = bar_timedelta(bar)
        self.buffer_days = buffer_days
        self.ex = exchange or BinanceSpot("https://data-api.binance.vision", "binance-data")
        self.kl: dict[str, pd.DataFrame] = {}

    def load(self, now: pd.Timestamp | None = None) -> None:
        now = now or pd.Timestamp.now(tz="UTC")
        start = (now - pd.Timedelta(days=self.buffer_days)).floor("D").tz_localize(None)
        with ThreadPoolExecutor(6) as pool:
            hist = dict(zip(self.symbols, pool.map(lambda s: self.md.klines(s, self.bar, start, market="spot"),
                                                   self.symbols)))
        self.kl = {s: h.drop(columns=[c for c in ("ignore", "close_time") if c in h]) for s, h in hist.items()}
        self.update()

    def update(self) -> None:
        def top_up(sym):
            h = self.kl.get(sym, pd.DataFrame())
            since = h.index[-1] + self.td if not h.empty else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
            add = self.ex.klines(sym, self.bar, since, full=True)
            if add.empty:
                return sym, h
            out = pd.concat([h, add[[c for c in h.columns if c in add.columns] or add.columns]])
            out = out[~out.index.duplicated(keep="last")].sort_index()  # newest copy of a bar wins
            cutoff = out.index[-1] - pd.Timedelta(days=self.buffer_days)
            return sym, out[out.index >= cutoff]

        with ThreadPoolExecutor(6) as pool:
            for sym, frame in pool.map(top_up, self.symbols):
                self.kl[sym] = frame

    def last_common_close(self, with_lagging: bool = False, max_lag: pd.Timedelta = pd.Timedelta(minutes=15)):
        """Close time of the newest bar every *active* symbol has.

        A symbol more than ``max_lag`` behind the freshest one (halted, delisted,
        API trouble) is excluded so it cannot freeze the whole book; it simply has
        no new bars (no new view, positions marked at its last price).
        """
        last = {s: k.index[-1] + self.td for s, k in self.kl.items() if not k.empty}
        newest = max(last.values())
        lagging = sorted(s for s, t in last.items() if newest - t > max_lag)
        t = min(v for s, v in last.items() if s not in lagging)
        return (t, lagging) if with_lagging else t

    def data(self, until: pd.Timestamp | None = None) -> IntradayData:
        until = until or self.last_common_close()
        kl = {s: k[k.index + self.td <= until] for s, k in self.kl.items()}
        return assemble_intraday(kl, {}, {}, self.bar, 0.0)

    def quotes(self) -> dict[str, Quote]:
        return self.ex.quotes(self.symbols)


def okx_funding_rates(symbols: list[str], session=None) -> dict[str, tuple[float, pd.Timestamp | None]]:
    """Live perpetual funding rates from OKX USDT swaps → {symbol: (rate, settlement time it applies to)}."""
    import requests

    s = session or requests.Session()
    out = {}
    for sym in symbols:
        base = sym.removesuffix("USDT")
        try:
            r = s.get(f"{OKX}/api/v5/public/funding-rate", params={"instId": f"{base}-USDT-SWAP"}, timeout=10)
            row = r.json()["data"][0]
            settle = pd.Timestamp(int(row["fundingTime"]), unit="ms", tz="UTC") if row.get("fundingTime") else None
            out[sym] = (float(row["fundingRate"]), settle)
        except Exception as exc:  # noqa: BLE001 - keep last known / default
            log.debug("funding rate %s unavailable: %s", sym, exc)
        time.sleep(0.12)
    return out


@dataclass
class IntradayCycle:
    now: pd.Timestamp
    bar: pd.Timestamp | None = None
    new_bars: int = 0
    traded: bool = False
    orders: int = 0
    funding_events: int = 0
    warnings: list[str] = field(default_factory=list)


class IntradayEngine:
    def __init__(self, cfg: LabConfig, feed: IntradayFeed, store: StateStore, models_dir: Path,
                 policies: list[PolicySpec] = DEFAULT_POLICIES, stale_minutes: float = 12.0):
        self.cfg = cfg
        self.ic = cfg.intraday
        self.feed = feed
        self.store = store
        self.models_dir = Path(models_dir)
        self.specs = policies
        self.stale = pd.Timedelta(minutes=stale_minutes)
        self.broker = PaperBroker(store, CostConfig(fee_bps=self.ic.fee_bps, slippage_bps=1.0), "perp")
        self.models: dict[str, object] = {}
        self.manifest = json.loads((self.models_dir / "manifest.json").read_text())
        trained_syms = self.manifest.get("symbols")
        if trained_syms is not None and list(trained_syms) != list(feed.symbols):
            raise RuntimeError(f"model symbols/order {trained_syms} != live feed {feed.symbols} "
                               "(asset ids and embeddings would be misaligned)")
        self.funding: dict[str, tuple[float, pd.Timestamp | None]] = {}
        self.funding_fetched: pd.Timestamp | None = None

    # ------------------------------------------------------------- models
    def _model(self, spec: PolicySpec):
        key = spec.model if spec.model == "deep" else f"lgbm_h{spec.horizon}"
        if key not in self.models:
            path = self.models_dir / key
            if spec.model == "deep":
                from tradinglab.intraday.deep import DeepIntradayTrader

                self.models[key] = DeepIntradayTrader(None).load(path)
            else:
                from tradinglab.intraday.models import LGBMForecaster

                self.models[key] = LGBMForecaster().load(path)
        return self.models[key]

    def _outputs(self, spec: PolicySpec, fs, d: IntradayData, rows: np.ndarray) -> np.ndarray:
        """Model output for each bar index in ``rows`` → [len(rows), N]."""
        names = self.manifest["features"]
        if names != fs.names:
            raise RuntimeError(f"feature mismatch: model {names} vs live {fs.names}")
        model = self._model(spec)
        N = len(fs.symbols)
        if spec.model == "deep":
            from tradinglab.intraday.deep import prepare

            dd = prepare(fs, d, model.cfg)
            return model.predict(dd, rows)
        X = fs.X[rows].reshape(len(rows) * N, -1)
        assets = np.tile(np.arange(N), len(rows))
        ok = np.isfinite(X).all(axis=1)
        out = np.full(len(rows) * N, np.nan)
        if ok.any():
            out[ok] = model.predict(X[ok], asset=assets[ok])
        return out.reshape(len(rows), N)

    # ----------------------------------------------------------- accounts
    def account(self, spec: PolicySpec) -> str:
        return f"intraday:{spec.name}"

    def ensure_accounts(self, now: pd.Timestamp) -> None:
        with self.store.transaction():
            for spec in self.specs:
                name = self.account(spec)
                if self.store.account(name) is None:
                    self.store.ensure_account(name, spec.description or spec.name, self.ic.initial_capital, now)
                    self.store.log_event(now, "info", f"account created: {spec.description}", name)

    def _policy(self, spec: PolicySpec):
        return make_policy(spec, self.feed.symbols, self.store.get_meta(f"policy:{self.account(spec)}"))

    # --------------------------------------------------------------- cycle
    def cycle(self, now: pd.Timestamp | None = None) -> IntradayCycle:
        """Process every bar that closed since the last cycle – atomically.

        Policy steps, targets and quotes are computed in memory first; marks,
        funding, policy state, executions, decisions and the ``last_bar`` marker
        are then committed in ONE transaction. Any failure (model, quotes, DB)
        rolls everything back and the next cycle redoes the same bars from the
        same persisted state – a bar is never applied twice and never skipped.
        """
        now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
        rep = IntradayCycle(now=now)
        self.feed.update()
        t_last, lagging = self.feed.last_common_close(with_lagging=True)
        rep.bar = t_last
        if lagging:
            rep.warnings.append(f"lagging symbols excluded: {', '.join(lagging)}")
        stale = now - t_last > self.stale
        if stale:
            rep.warnings.append(f"latest bar {t_last} is stale at {now}; not trading")
        last_done = self.store.get_meta("intraday:last_bar")
        last_done = utc(last_done) if last_done else None
        if last_done is not None and t_last <= last_done:
            return rep
        self.ensure_accounts(now)
        d = self.feed.data(t_last)
        fs = build_features(d, 12)
        idx = d.index
        new = idx[-1:] if last_done is None else idx[(idx > last_done) & (idx <= t_last)]
        rep.new_bars = len(new)
        rows = np.searchsorted(idx, new)
        self._refresh_funding(now)

        targets: dict[str, pd.Series] = {}
        states: dict[str, dict] = {}
        for spec in self.specs:
            pol = self._policy(spec)
            if spec.rule == "selective" and not pol.preds:
                self._warm_up(spec, pol, fs, d, rows[0])
            done_until = utc(pol.last_bar) if pol.last_bar else None
            out = self._outputs(spec, fs, d, rows)
            for k, r in enumerate(rows):
                if done_until is not None and idx[r] <= done_until:
                    log.warning("%s: bar %s already applied to policy state; skipping", spec.name, idx[r])
                    continue
                last = k == len(rows) - 1
                w = pol.update(out[k], idx[r], d, self.ic, size=last and not stale)
                if w is not None:
                    targets[spec.name] = w
            states[spec.name] = pol.state()
        quotes = self.feed.quotes() if targets else {}

        with self.store.transaction():
            for spec in self.specs:
                self._mark_and_fund(spec, d, new, last_done, rep)
                self.store.set_meta(f"policy:{self.account(spec)}", states[spec.name])
            if targets:
                self._execute(targets, quotes, t_last, now, rep)
            self.store.set_meta("intraday:last_bar", str(t_last))
        self._export(rep)
        return rep

    def _warm_up(self, spec: PolicySpec, pol, fs, d: IntradayData, upto: int) -> None:
        """Rebuild forecast history over the buffer (no trading) so scores are normalised from bar one."""
        lo = max(0, upto - spec.norm_window - spec.horizon)
        rows = np.arange(lo, upto)
        out = self._outputs(spec, fs, d, rows)
        for k, r in enumerate(rows):
            pol.update(out[k], d.index[r], size=False)
        log.info("%s warmed up on %d bars", spec.name, len(rows))

    def _refresh_funding(self, now: pd.Timestamp) -> None:
        """Refresh live funding rates at most every 30 min (also after failures – no per-bar retries)."""
        if self.funding_fetched is None or now - self.funding_fetched > pd.Timedelta(minutes=30):
            self.funding_fetched = now
            try:
                rates = okx_funding_rates(self.feed.symbols)
            except Exception as exc:  # noqa: BLE001
                log.warning("funding refresh failed: %s", exc)
                return
            for sym, (rate, settle) in rates.items():
                self.funding[sym] = (rate, settle)

    def _funding_rate(self, sym: str, stamp: pd.Timestamp) -> tuple[float, bool]:
        """Rate for the settlement at ``stamp`` → (rate, estimated)."""
        cached = self.funding.get(sym)
        if cached is None:
            return 0.0001, True
        rate, settle = cached
        return rate, not (settle is not None and settle == stamp)

    def _mark_and_fund(self, spec: PolicySpec, d: IntradayData, new: pd.DatetimeIndex,
                       last_done: pd.Timestamp | None, rep: IntradayCycle) -> None:
        name = self.account(spec)
        closes = d.close.ffill()  # last known price for symbols missing a bar (gaps, lagging feeds)
        prev = last_done
        for t in new:
            prices = {s: float(p) for s, p in closes.loc[t].items() if np.isfinite(p)}
            if prev is not None:
                for stamp in pd.date_range(prev.ceil("8h"), t, freq="8h"):
                    if prev < stamp <= t:
                        for sym in self.store.positions(name):
                            if sym not in prices:
                                continue
                            rate, est = self._funding_rate(sym, stamp)
                            if self.broker.apply_funding(name, sym, stamp, rate, prices[sym], est):
                                rep.funding_events += 1
            eq, gross, net = self.broker.equity(name, prices)
            cash = self.store.account(name)["cash"]
            self.store.record_equity(name, t, eq, cash, gross / eq if eq else 0.0, net / eq if eq else 0.0)
            prev = t

    def _execute(self, targets: dict[str, pd.Series], quotes: dict[str, Quote], t_last: pd.Timestamp,
                 now: pd.Timestamp, rep: IntradayCycle) -> None:
        """Rebalance every account at the live quotes (caller holds the transaction)."""
        mids = {s: q.mid for s, q in quotes.items()}
        for spec in self.specs:
            if spec.name not in targets:
                continue
            name = self.account(spec)
            eq, _, _ = self.broker.equity(name, mids)
            guard = DrawdownGuard(self.cfg.risk)
            guard.restore(self.store.get_meta(f"guard:{name}", {}))
            scale = guard.update(eq)
            tgt = {s: float(w) * scale for s, w in targets[spec.name].items() if abs(w) > 1e-12}
            ids = self.broker.rebalance(name, tgt, mids, now, now, self.ic.rebalance_band,
                                        self.ic.min_trade_notional, reason=f"bar {t_last}")
            for o in self.store.pending_orders(name):
                q = quotes.get(o["symbol"])
                if q is not None:
                    self.broker.fill(o, q.mid, now, quote=q)
                else:
                    self.store.cancel_order(o["id"], "no live quote")
            self.store.record_decision(name, t_last, now, eq, scale, tgt)
            self.store.set_meta(f"guard:{name}", guard.state())
            rep.orders += len(ids)
        rep.traded = True

    # -------------------------------------------------------------- status
    def status(self) -> dict:
        out = {}
        for spec in self.specs:
            name = self.account(spec)
            acct = self.store.account(name)
            if acct is None:
                continue
            last = self.store.last_equity(name)
            eq = last["equity"] if last else acct["initial_capital"]
            orders = self.store.orders(name, limit=100000)
            filled = orders[orders["status"] == "filled"] if not orders.empty else orders
            out[name] = {
                "model": spec.description, "equity": eq, "pnl": eq - acct["initial_capital"],
                "return": eq / acct["initial_capital"] - 1.0, "marked_at": last["ts"] if last else None,
                "gross": last["gross"] if last else 0.0, "net": last["net"] if last else 0.0,
                "trades": int(len(filled)), "fees_paid": float(filled["fee"].sum()) if len(filled) else 0.0,
                "funding_paid": -self.store.funding_total(name), "positions": self.store.positions(name),
            }
        return out

    def _export(self, rep: IntradayCycle) -> None:
        state_dir = Path(self.cfg.paths.state_dir)
        status = {"updated_at": str(rep.now), "last_bar": str(rep.bar), "new_bars": rep.new_bars,
                  "traded": rep.traded, "orders": rep.orders, "warnings": rep.warnings,
                  "models": self.manifest.get("models"), "accounts": self.status()}
        (state_dir / "intraday_status.json").write_text(json.dumps(status, indent=2, default=str))


def run_intraday(cfg: LabConfig, poll_seconds: float = 10.0, max_cycles: int | None = None,
                 commit_every_minutes: float | None = None) -> None:
    """Daemon loop: process each new 5-minute bar as soon as it closes; never die."""
    from tradinglab.live.daemon import commit_state

    state_dir = Path(cfg.paths.state_dir)
    store = StateStore(state_dir / "intraday.sqlite")
    feed = IntradayFeed(MarketData(cfg.paths.data_dir), cfg.intraday.symbols)
    feed.load()
    engine = IntradayEngine(cfg, feed, store, Path(cfg.paths.models_dir) / "intraday" / "live")
    cycles, backoff, last_commit = 0, 5.0, time.time()
    while True:
        try:
            rep = engine.cycle()
            if rep.new_bars:
                st = engine.status()
                log.info("bar %s: orders=%d warnings=%s | %s", rep.bar, rep.orders, rep.warnings,
                         " ".join(f"{k.split(':')[1]}={v['pnl']:+.2f}" for k, v in st.items()))
            (state_dir / "intraday_heartbeat.json").write_text(json.dumps({"ts": str(pd.Timestamp.now(tz='UTC')),
                                                                          "last_bar": str(rep.bar)}))
            backoff = 5.0
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            log.exception("intraday cycle failed: %s", exc)
            time.sleep(min(backoff, 300))
            backoff *= 2
            continue
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        if commit_every_minutes and time.time() - last_commit > commit_every_minutes * 60:
            commit_state(state_dir, f"intraday: live state {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}")
            last_commit = time.time()
        time.sleep(poll_seconds)
