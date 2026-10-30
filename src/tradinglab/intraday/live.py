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

import copy
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
from tradinglab.intraday.attribution import (attribute, bar_costs, bar_flows, basket_returns, held_units,
                                             start_exposure)
from tradinglab.intraday.data import IntradayData, assemble_intraday
from tradinglab.intraday.features import build_features
from tradinglab.intraday.policy import (DEFAULT_POLICIES, ContinuousPolicy, HysteresisBook, PolicySpec,
                                       SelectiveScorer, policies_by_name)
from tradinglab.live.broker import PaperBroker
from tradinglab.live.state import StateStore
from tradinglab.live.vault import VaultRule, bank, vault_accounts
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
    banked: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class IntradayAccount:
    """A paper account: one strategy, optionally with a profit vault."""

    name: str                    # "intraday:lgbm_4h" or "intraday:lgbm_4h+vault2"
    spec: PolicySpec
    vault: VaultRule | None = None


def build_accounts(specs: list[PolicySpec], thresholds: list[float] | None = None) -> list[IntradayAccount]:
    """Every strategy gets a plain account plus one vault twin per threshold."""
    out = []
    for spec in specs:
        for name, _, rule in vault_accounts([f"intraday:{spec.name}"], list(thresholds or [])):
            out.append(IntradayAccount(name, spec, rule))
    return out


_BAR_RE = r"^bar (\d{4}-\d\d-\d\d[ T][0-9:.]+(?:[+-]\d\d:\d\d|Z)?)"


class IntradayEngine:
    def __init__(self, cfg: LabConfig, feed: IntradayFeed, store: StateStore, models_dir: Path,
                 policies: list[PolicySpec] = DEFAULT_POLICIES, stale_minutes: float = 12.0,
                 vault_thresholds: list[float] | None = None):
        self.cfg = cfg
        self.ic = cfg.intraday
        self.feed = feed
        self.store = store
        self.models_dir = Path(models_dir)
        self.specs = list(policies)
        thresholds = self.ic.vault_thresholds if vault_thresholds is None else vault_thresholds
        self.accounts = build_accounts(self.specs, thresholds)
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
        # Selective scores are a pure function of the model's forecast history, shared by every account
        # trading that forecast. They live in memory only and are rebuilt from the price buffer after a
        # restart (warm-up), which keeps the persisted state small; accounts persist just their book.
        self.scorers: dict[str, SelectiveScorer] = {}

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

    def _features(self, d: IntradayData):
        """Model features over the price buffer (the simulator supplies precomputed signals instead)."""
        return build_features(d, 12)

    def _outputs(self, spec: PolicySpec, fs, d: IntradayData, rows: np.ndarray) -> np.ndarray:
        """Model (or rule) output for each bar index in ``rows`` → [len(rows), N]."""
        N = len(fs.symbols)
        if spec.model.startswith("rule:"):
            return rule_outputs(spec.model[5:], d, rows)
        names = self.manifest["features"]
        if names != fs.names:
            raise RuntimeError(f"feature mismatch: model {names} vs live {fs.names}")
        model = self._model(spec)
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
        """The plain (no-vault) account of a strategy."""
        return f"intraday:{spec.name}"

    def ensure_accounts(self, now: pd.Timestamp) -> None:
        with self.store.transaction():
            for acct in self.accounts:
                if self.store.account(acct.name) is None:
                    label = acct.spec.description or acct.spec.name
                    if acct.vault is not None:
                        label += f" | vault: bank profit at +{100 * acct.vault.threshold:g}%"
                    self.store.ensure_account(acct.name, label, self.ic.initial_capital, now)
                    self.store.log_event(now, "info", f"account created: {label}", acct.name)

    def _book(self, acct: IntradayAccount):
        state = self.store.get_meta(f"policy:{acct.name}")
        if acct.spec.rule == "selective":
            return HysteresisBook(acct.spec, self.feed.symbols, state)
        return ContinuousPolicy(acct.spec, self.feed.symbols, state)

    def cycle(self, now: pd.Timestamp | None = None) -> IntradayCycle:
        """Process every bar that closed since the last cycle – atomically.

        Signals, policy steps, targets and quotes are computed in memory first;
        marks, funding, profit banking, policy state, executions, decisions and
        the ``last_bar`` marker are then committed in ONE transaction. Any failure
        (model, quotes, DB) rolls everything back – including the in-memory
        scorers – and the next cycle redoes the same bars from the same persisted
        state: a bar is never applied twice and never skipped.
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
        fs = self._features(d)
        idx = d.index
        new = idx[-1:] if last_done is None else idx[(idx > last_done) & (idx <= t_last)]
        rep.new_bars = len(new)
        rows = np.searchsorted(idx, new)
        self._refresh_funding(now)

        # 1. signals, once per signal source
        outputs: dict[str, np.ndarray] = {}
        scores: dict[str, list[np.ndarray | None]] = {}
        scorers: dict[str, SelectiveScorer] = {}
        for spec in self.specs:
            key = spec.signal_key
            if key in outputs:
                continue
            outputs[key] = self._outputs(spec, fs, d, rows)
            if spec.rule == "selective":
                sc = copy.deepcopy(self.scorers.get(key)) if key in self.scorers else None
                if sc is None:
                    sc = SelectiveScorer(spec.horizon, spec.norm_window)
                    self._warm_up(spec, sc, fs, d, rows[0])
                done = utc(sc.last_bar) if sc.last_bar else None
                scores[key] = [None if done is not None and idx[r] <= done else sc.score(outputs[key][k], idx[r])
                               for k, r in enumerate(rows)]
                scorers[key] = sc

        # 2. each account advances its own book; continuous targets are shared per strategy
        targets: dict[str, pd.Series] = {}
        books: dict[str, object] = {}
        shared: dict[str, pd.Series | None] = {}
        for acct in self.accounts:
            spec, key = acct.spec, acct.spec.signal_key
            book = self._book(acct)
            done_until = utc(book.last_bar) if book.last_bar else None
            for k, r in enumerate(rows):
                if done_until is not None and idx[r] <= done_until:
                    log.warning("%s: bar %s already applied to policy state; skipping", acct.name, idx[r])
                    continue
                size = k == len(rows) - 1 and not stale
                if spec.rule == "selective":
                    if scores[key][k] is None:
                        continue
                    w = book.update(scores[key][k], idx[r], d, self.ic, size=size)
                else:
                    if size and spec.name not in shared:
                        shared[spec.name] = book.update(outputs[key][k], idx[r], d, self.ic, size=True)
                    book.last_bar = str(idx[r])
                    w = shared.get(spec.name) if size else None
                if w is not None:
                    targets[acct.name] = w
            books[acct.name] = book
        quotes = self.feed.quotes() if not stale else {}

        with self.store.transaction():
            for acct in self.accounts:
                self._mark_and_fund(acct.name, d, new, last_done, rep)
            banked = set()
            if not stale:
                for acct in self.accounts:
                    if acct.vault is not None and self._maybe_bank(acct, quotes, t_last, now, rep):
                        banked.add(acct.name)
                        books[acct.name].reset()
            for acct in self.accounts:
                self.store.set_meta(f"policy:{acct.name}", books[acct.name].state())
            todo = {n: w for n, w in targets.items() if n not in banked}
            if todo:
                self._execute(todo, quotes, t_last, now, rep)
            self.store.set_meta("intraday:last_bar", str(t_last))
        self.scorers.update(scorers)
        self._export(rep, d.close)
        return rep

    def _warm_up(self, spec: PolicySpec, sc: SelectiveScorer, fs, d: IntradayData, upto: int) -> None:
        """Rebuild the forecast history over the buffer (no trading) so scores are normalised from bar one."""
        lo = max(0, upto - spec.norm_window - spec.horizon)
        rows = np.arange(lo, upto)
        if len(rows):
            out = self._outputs(spec, fs, d, rows)
            for k, r in enumerate(rows):
                sc.score(out[k], d.index[r])
        log.info("%s scorer warmed up on %d bars", spec.signal_key, len(rows))

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

    def _mark_and_fund(self, name: str, d: IntradayData, new: pd.DatetimeIndex,
                       last_done: pd.Timestamp | None, rep: IntradayCycle) -> None:
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

    def _close_all(self, name: str, quotes: dict[str, Quote], t_last: pd.Timestamp, now: pd.Timestamp,
                   why: str) -> int | None:
        """Close every position at the live quotes; None (nothing done) if a quote is missing."""
        pos = self.store.positions(name)
        if any(sym not in quotes for sym in pos):
            return None
        quotes = {s: quotes[s] for s in pos}
        mids = {s: q.mid for s, q in quotes.items()}
        ids = self.broker.rebalance(name, {}, mids, now, now, 0.0, 0.0, reason=f"bar {t_last} ({why})")
        for o in self.store.pending_orders(name):
            q = quotes[o["symbol"]]
            self.broker.fill(o, q.mid, q.ts, quote=q)
        return len(ids)

    def _maybe_bank(self, acct: IntradayAccount, quotes: dict[str, Quote], t_last: pd.Timestamp,
                    now: pd.Timestamp, rep: IntradayCycle) -> bool:
        """Vault rule at the bar-t mark: realise everything and bank the excess over the base."""
        last = self.store.last_equity(acct.name)
        if last is None or utc(last["ts"]) != t_last:
            return False
        eq, base = last["equity"], self.store.account(acct.name)["initial_capital"]
        if not acct.vault.due(eq, base):
            return False
        n = self._close_all(acct.name, quotes, t_last, now, "vault: realise profit")
        if n is None:
            rep.warnings.append(f"{acct.name}: vault due but a quote is missing; retry next bar")
            return False
        amount = bank(self.store, acct.name, now, eq, note=f"threshold {acct.vault.threshold:g}", mark_ts=t_last)
        self.store.set_meta(f"guard:{acct.name}", {})       # new high-water mark from the base
        self.store.record_decision(acct.name, t_last, now, eq, 1.0, {})
        rep.orders += n
        rep.banked += amount
        return True

    def _execute(self, targets: dict[str, pd.Series], quotes: dict[str, Quote], t_last: pd.Timestamp,
                 now: pd.Timestamp, rep: IntradayCycle) -> None:
        """Rebalance accounts at the live quotes (caller holds the transaction)."""
        mids = {s: q.mid for s, q in quotes.items()}
        for name, w in targets.items():
            eq, _, _ = self.broker.equity(name, mids)
            guard = DrawdownGuard(self.cfg.risk)
            guard.restore(self.store.get_meta(f"guard:{name}", {}))
            scale = guard.update(eq)
            tgt = {s: float(x) * scale for s, x in w.items() if abs(x) > 1e-12}
            ids = self.broker.rebalance(name, tgt, mids, now, now, self.ic.rebalance_band,
                                        self.ic.min_trade_notional, reason=f"bar {t_last}")
            for o in self.store.pending_orders(name):
                q = quotes.get(o["symbol"])
                if q is not None:
                    self.broker.fill(o, q.mid, q.ts, quote=q)  # stamped when the quote was taken
                else:
                    self.store.cancel_order(o["id"], "no live quote")
            if ids:  # decisions are journaled when they trade (targets are recomputable from the data)
                self.store.record_decision(name, t_last, now, eq, scale, tgt)
            self.store.set_meta(f"guard:{name}", guard.state())
            rep.orders += len(ids)
        rep.traded = True

    def liquidate(self, now: pd.Timestamp | None = None, reason: str = "experiment end") -> dict[str, float]:
        """Close every account at live quotes and mark the final equity (= cash)."""
        now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
        quotes = dict(self.feed.quotes())
        last = self.store.get_meta("intraday:last_bar")
        t_last = utc(last) if last else now
        # A coin without a live quote is closed at its last known price (logged) rather than blocking the end.
        closes = self.feed.data().close.ffill().iloc[-1]
        for sym in self.feed.symbols:
            if sym not in quotes and np.isfinite(closes.get(sym, np.nan)):
                px = float(closes[sym])
                log.warning("liquidate: no live quote for %s, using last close %.6g", sym, px)
                quotes[sym] = Quote(px, px, now, "last-close")
        out = {}
        with self.store.transaction():
            for acct in self.accounts:
                if self.store.account(acct.name) is None:
                    continue
                if self._close_all(acct.name, quotes, t_last, now, reason) is None:
                    raise RuntimeError(f"{acct.name}: cannot liquidate, no price for an open position")
                cash = self.store.account(acct.name)["cash"]
                self.store.record_equity(acct.name, now, cash, cash, 0.0, 0.0)
                self.store.log_event(now, "info", f"liquidated ({reason})", acct.name)
                out[acct.name] = cash
        self._export(IntradayCycle(now=now, bar=t_last, warnings=[f"liquidated: {reason}"]))
        return out

    # -------------------------------------------------------------- status
    def status(self, close: pd.DataFrame | None = None) -> dict:
        """Account summaries; with bar ``close`` prices also the market/skill P&L attribution."""
        out = {}
        for acct in self.accounts:
            name = acct.name
            row = self.store.account(name)
            if row is None:
                continue
            last = self.store.last_equity(name)
            eq = last["equity"] if last else row["initial_capital"]
            vault = self.store.vault_total(name)
            # Profit banked after the latest mark is still inside that mark's equity: count it once.
            vault_in_pnl = self.store.vault_total(name, before_mark=last["ts"]) if last else vault
            orders = self.store.orders(name, limit=10_000_000)
            filled = orders[orders["status"] == "filled"] if not orders.empty else orders
            out[name] = {
                "strategy": acct.spec.name, "vault_threshold": acct.vault.threshold if acct.vault else None,
                "model": acct.spec.description, "equity": eq, "vault": vault,
                "pnl": eq + vault_in_pnl - row["initial_capital"],
                "return": (eq + vault_in_pnl) / row["initial_capital"] - 1.0,
                "marked_at": last["ts"] if last else None,
                "gross": last["gross"] if last else 0.0, "net": last["net"] if last else 0.0,
                "trades": int(len(filled)), "fees_paid": float(filled["fee"].sum()) if len(filled) else 0.0,
                "funding_paid": -self.store.funding_total(name), "positions": self.store.positions(name),
                "sweeps": int(len(self.store.vault_events(name))),
            }
            if close is not None:
                try:
                    out[name]["attribution"] = self._attribution(name, filled, close)
                except Exception as exc:  # noqa: BLE001 - one account's report must not hide the others
                    log.warning("%s: attribution failed: %s", name, exc)
                    out[name]["attribution"] = None
        return out

    def _attribution(self, name: str, filled: pd.DataFrame, close: pd.DataFrame) -> dict | None:
        """P&L split (see ``intraday.attribution``) over the marks the price buffer covers for every
        symbol still trading (a halted symbol's stale buffer must not stretch the window)."""
        last = close.index[-1]
        lv = {s: close[s].last_valid_index() for s in close.columns}
        active = [s for s, t in lv.items() if t is not None and last - t <= pd.Timedelta(hours=1)]
        if not active:
            return None
        t0 = max(close[s].first_valid_index() for s in active)
        eq = self.store.equity_curve(name)
        if eq.empty:
            return None
        eq = eq[(eq.index >= t0) & (eq.index <= last)]
        if len(eq) < 3:
            return None
        fills = self._fills(name, filled, eq.index)
        units = held_units(fills, eq.index, list(close.columns))
        fund = pd.DataFrame([dict(r) for r in self.store.conn.execute(
            "SELECT ts, amount FROM funding WHERE account=?", (name,))], columns=["ts", "amount"])
        fund["ts"] = pd.to_datetime(fund["ts"], utc=True, format="ISO8601")
        vault = self.store.vault_events(name)
        flows = None
        if not vault.empty:
            vt = pd.to_datetime(vault["mark_ts"].fillna(vault["ts"]), utc=True, format="ISO8601")
            flows = pd.DataFrame({"ts": vt, "amount": vault["amount"]})  # keyed to the triggering mark
        return attribute(eq["equity"], start_exposure(units, close), basket_returns(close[active]),
                         bar_costs(fills, fund, eq.index), flows=bar_flows(flows, eq.index))

    def _fills(self, name: str, filled: pd.DataFrame, marks: pd.DatetimeIndex) -> pd.DataFrame:
        """Filled orders with their decision bar (orders carry ``reason = "bar <close time> ..."``;
        should one not parse, the last mark at or before its fill time stands in)."""
        cols = ["bar", "symbol", "qty", "fee", "slippage_cost"]
        if filled is None or not len(filled):
            return pd.DataFrame(columns=cols)
        f = filled.copy()
        f["bar"] = pd.to_datetime(f["reason"].astype(str).str.extract(_BAR_RE)[0], utc=True,
                                  format="ISO8601", errors="coerce")
        miss = f["bar"].isna()
        if miss.any():
            ft = pd.to_datetime(f.loc[miss, "fill_ts"], utc=True, format="ISO8601")
            k = np.searchsorted(marks.values, ft.values, side="right") - 1
            f.loc[miss, "bar"] = [marks[i] if i >= 0 else marks[0] - pd.Timedelta(days=1) for i in k]
        f["slippage_cost"] = f["slippage_cost"].fillna(0.0)
        return f[cols]

    def _export(self, rep: IntradayCycle, close: pd.DataFrame | None = None) -> None:
        state_dir = Path(self.cfg.paths.state_dir)
        try:
            accounts = self.status(close)
        except Exception as exc:  # noqa: BLE001 - reporting must never block trading
            log.warning("attribution failed: %s", exc)
            accounts = self.status()
        status = {"updated_at": str(rep.now), "last_bar": str(rep.bar), "new_bars": rep.new_bars,
                  "traded": rep.traded, "orders": rep.orders, "banked": rep.banked, "warnings": rep.warnings,
                  "models": self.manifest.get("models"), "accounts": accounts}
        (state_dir / "intraday_status.json").write_text(json.dumps(status, indent=2, default=str))


def rule_outputs(rule: str, d: IntradayData, rows: np.ndarray) -> np.ndarray:
    """Outputs of the rule-based strategies at bar indices ``rows`` (causal: bars ≤ t only).

    ``basket`` → constant long positions; ``tsmom<n>`` → the n-bar log return (a forecast that the
    selective rule normalises and trades with hysteresis)."""
    n = len(d.symbols)
    if rule == "basket":
        return np.ones((len(rows), n))
    if rule.startswith("tsmom"):
        lb = int(rule[5:])
        logc = np.log(d.close.ffill().to_numpy())
        out = np.full((len(rows), n), np.nan)
        ok = rows >= lb
        out[ok] = logc[rows[ok]] - logc[rows[ok] - lb]
        return out
    raise ValueError(f"unknown rule {rule!r}")


def run_intraday(cfg: LabConfig, poll_seconds: float = 10.0, max_cycles: int | None = None,
                 commit_every_minutes: float | None = None, until: pd.Timestamp | None = None) -> None:
    """Daemon loop: process each new 5-minute bar as soon as it closes; never die.

    Stops at ``until`` (or on SIGTERM/SIGINT) after the current step."""
    import signal as _signal

    from tradinglab.live.daemon import commit_state

    stop = {"flag": False}

    def _handle(signum, _frame):
        log.info("signal %s received – stopping after current step", signum)
        stop["flag"] = True

    _signal.signal(_signal.SIGTERM, _handle)
    _signal.signal(_signal.SIGINT, _handle)
    state_dir = Path(cfg.paths.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    store = StateStore(state_dir / "intraday.sqlite")
    feed = IntradayFeed(MarketData(cfg.paths.data_dir), cfg.intraday.symbols)
    feed.load()
    engine = IntradayEngine(cfg, feed, store, Path(cfg.paths.models_dir) / "intraday" / "live",
                            policies=policies_by_name(cfg.intraday.live_policies))
    cycles, backoff, last_commit = 0, 5.0, time.time()
    while not stop["flag"] and (until is None or pd.Timestamp.now(tz="UTC") < until):
        try:
            rep = engine.cycle()
            if rep.new_bars:
                log.info("bar %s: new=%d orders=%d banked=%.2f warnings=%s", rep.bar, rep.new_bars, rep.orders,
                         rep.banked, rep.warnings)
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
        for _ in range(int(max(poll_seconds, 1))):
            if stop["flag"]:
                break
            time.sleep(1)
    store.close()
