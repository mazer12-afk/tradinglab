"""Autonomous live paper-trading engine.

One :meth:`LiveEngine.cycle` is idempotent and safe to run at any frequency
(cron, daemon loop, CI schedule). Each cycle:

1. refreshes market data (realtime exchanges or the public archive);
2. replays, in time order, every event that became knowable since the last
   cycle – order fills, perpetual funding payments and hourly marks – so the
   equity curve is exact even when data arrives late;
3. when a new signal bar is complete (and the configured execution delay has
   elapsed) asks every account's strategy for target weights, applies the
   drawdown guard and submits rebalancing orders timestamped *now*;
4. persists everything to SQLite and exports a human-readable status.

Accounts: ``main`` trades the research-selected ensemble; ``shadow:<name>``
accounts paper-trade every candidate so the lab keeps measuring live performance
of the whole zoo (a live tournament).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.config import LabConfig, bars_per_year, utc
from tradinglab.live.broker import PaperBroker
from tradinglab.live.feed import LiveFeed, latest_bar
from tradinglab.live.state import StateStore
from tradinglab.live.vault import VaultRule, bank
from tradinglab.portfolio.allocator import combine_weights
from tradinglab.portfolio.risk import DrawdownGuard
from tradinglab.portfolio.sizing import strategy_weights
from tradinglab.strategies.base import Strategy
from tradinglab.strategies.registry import by_name

log = logging.getLogger(__name__)

MAIN = "main"


@dataclass
class AccountSpec:
    name: str
    strategies: list[str]
    alphas: dict[str, float]
    vault: VaultRule | None = None

    @property
    def label(self) -> str:
        return "+".join(self.strategies) if self.strategies else "flat"


@dataclass
class CycleReport:
    now: pd.Timestamp
    mode: str
    horizon: pd.Timestamp | None = None
    signal_bar: pd.Timestamp | None = None
    fills: int = 0
    funding_events: int = 0
    marks: int = 0
    decisions: list[str] = field(default_factory=list)
    orders: int = 0
    banked: float = 0.0
    warnings: list[str] = field(default_factory=list)


def load_selection(cfg: LabConfig) -> dict:
    path = Path(cfg.paths.models_dir) / "selection.json"
    if not path.exists():
        return {"strategies": [], "alphas": {}, "candidates": [], "deep_models": {}}
    return json.loads(path.read_text())


class LiveEngine:
    def __init__(self, cfg: LabConfig, feed: LiveFeed, store: StateStore, selection: dict | None = None):
        self.cfg = cfg
        self.feed = feed
        self.store = store
        self.broker = PaperBroker(store, cfg.costs, cfg.market.instrument)
        self.selection = selection if selection is not None else load_selection(cfg)
        self.bpy = bars_per_year(cfg.market.bar)
        self.specs = self._account_specs()
        self._strategies: dict[str, Strategy] = {}

    # ------------------------------------------------------------ accounts
    def _account_specs(self) -> list[AccountSpec]:
        sel = self.selection
        specs = [AccountSpec(MAIN, list(sel.get("strategies", [])), dict(sel.get("alphas", {})))]
        if self.cfg.live.shadow_all:
            for name in sel.get("candidates", []):
                if name.startswith("dl_") and name not in sel.get("deep_models", {}):
                    continue
                specs.append(AccountSpec(f"shadow:{name}", [name], {name: 1.0}))
        twins = []
        for spec in specs:
            twins.append(spec)
            for t in self.cfg.live.vault_thresholds:
                rule = VaultRule(float(t))
                twins.append(AccountSpec(spec.name + rule.suffix, spec.strategies, spec.alphas, rule))
        return twins

    def ensure_accounts(self, now: pd.Timestamp) -> None:
        with self.store.transaction():
            for spec in self.specs:
                if self.store.account(spec.name) is None:
                    label = spec.label + (f" | vault: bank profit at +{100 * spec.vault.threshold:g}%"
                                          if spec.vault else "")
                    self.store.ensure_account(spec.name, label, self.cfg.portfolio.initial_capital, now)
                    self.store.set_meta(f"last_mark:{spec.name}", str(now.floor("h")))
                    self.store.log_event(now, "info", f"account created ({spec.label})", spec.name)

    def strategy(self, name: str) -> Strategy:
        if name not in self._strategies:
            s = by_name(self.cfg, name)
            if s.trainable:
                path = self.selection.get("deep_models", {}).get(name)
                if not path:
                    raise RuntimeError(f"no trained live model for {name}")
                s.load(Path(path))
            self._strategies[name] = s
        return self._strategies[name]

    # -------------------------------------------------------------- events
    def _advance(self, account: str, horizon: pd.Timestamp, rep: CycleReport) -> None:
        """Apply fills, funding and hourly marks in (last_mark, horizon] in time order."""
        last_mark = utc(self.store.get_meta(f"last_mark:{account}"))
        if horizon <= last_mark:
            return
        events: list[tuple] = []
        for o in self.store.pending_orders(account):
            t = utc(o["fill_after_ts"])
            if t <= horizon:
                events.append((t, 0, "fill", o))
        pos_symbols = set(self.store.positions(account)) | {o["symbol"] for o in self.store.pending_orders(account)}
        if self.cfg.market.instrument == "perp":
            for sym in pos_symbols:
                for stamp, rate, est in self.feed.funding_events(sym, last_mark, horizon):
                    events.append((stamp, 1, "funding", (sym, rate, est)))
        for h in pd.date_range(last_mark + pd.Timedelta(hours=1), horizon, freq="h"):
            events.append((h, 2, "mark", None))
        events.sort(key=lambda e: (e[0], e[1]))
        for t, _, kind, payload in events:
            if kind == "fill":
                px = self.feed.fill_price(payload["symbol"], t)
                if px is None:
                    if t < horizon - pd.Timedelta(days=2):
                        self.store.cancel_order(payload["id"], "no fill price available")
                        self.store.log_event(t, "warning", f"cancelled order {payload['id']} (no price)", account)
                    continue
                self.broker.fill(payload, px, t)
                rep.fills += 1
            elif kind == "funding":
                sym, rate, est = payload
                price = self.feed.price_at(sym, t)
                if price is not None and self.broker.apply_funding(account, sym, t, rate, price, est):
                    rep.funding_events += 1
            else:
                prices = self._prices(account, t)
                eq, gross, net = self.broker.equity(account, prices)
                acct = self.store.account(account)
                self.store.record_equity(account, t, eq, acct["cash"], gross / eq if eq else 0.0,
                                         net / eq if eq else 0.0)
                rep.marks += 1
        self.store.set_meta(f"last_mark:{account}", str(horizon))

    def _prices(self, account: str, t: pd.Timestamp) -> dict[str, float]:
        out = {}
        for sym in self.store.positions(account):
            p = self.feed.price_at(sym, t)
            if p is not None:
                out[sym] = p
        return out

    # ----------------------------------------------------------- decisions
    def _target_weights(self, spec: AccountSpec, panel, cache: dict) -> pd.Series:
        if not spec.strategies:
            return pd.Series(0.0, index=panel.close.columns)
        weights = {}
        for name in spec.strategies:
            if name not in cache:
                strat = self.strategy(name)
                if strat.trainable:
                    sig = strat.signals(panel, start=panel.close.index[-1])
                else:
                    sig = strat.signals(panel)
                cache[name] = strategy_weights(strat, sig, panel, self.cfg)
            weights[name] = cache[name]
        combined = combine_weights(weights, spec.alphas, panel, self.cfg)
        return combined.iloc[-1].fillna(0.0)

    def _guard(self, account: str) -> DrawdownGuard:
        g = DrawdownGuard(self.cfg.risk)
        g.restore(self.store.get_meta(f"guard:{account}", {}))
        return g

    @property
    def halt_file(self) -> Path:
        return Path(self.cfg.paths.state_dir) / "HALT"

    def _decide(self, spec: AccountSpec, panel, bar_ts: pd.Timestamp, horizon: pd.Timestamp,
                now: pd.Timestamp, cache: dict, rep: CycleReport) -> None:
        account = spec.name
        targets = self._target_weights(spec, panel, cache)
        if self.halt_file.exists():
            # Operator kill switch: flatten everything, take no new risk.
            targets = targets * 0.0
        # Size with the freshest known prices (== bar close in archive mode).
        prices = {}
        for s in set(targets.index) | set(self.store.positions(account)):
            p = self.feed.price_at(s, horizon)
            if p is not None and np.isfinite(p):
                prices[s] = p
        eq, _, _ = self.broker.equity(account, prices)
        guard = self._guard(account)
        scale = guard.update(eq)
        targets = targets * scale
        fill_after = now + pd.Timedelta(seconds=self.cfg.live.fill_latency_seconds)
        tdict = {s: float(w) for s, w in targets.items() if abs(w) > 1e-9}
        with self.store.transaction():
            ids = self.broker.rebalance(account, tdict, prices, now, fill_after, self.cfg.portfolio.rebalance_band,
                                        self.cfg.portfolio.min_trade_notional, reason=f"bar {bar_ts}")
            self.store.record_decision(account, bar_ts, now, eq, scale, tdict)
            self.store.set_meta(f"guard:{account}", guard.state())
            if scale < 1.0:
                self.store.log_event(now, "warning", f"drawdown guard scale {scale:.2f}", account)
        rep.decisions.append(account)
        rep.orders += len(ids)
        if self.feed.mode == "realtime" and ids:
            self._fill_now(account, now)

    def _fill_now(self, account: str, now: pd.Timestamp) -> None:
        """Realtime mode: fill at the current touch immediately."""
        with self.store.transaction():
            for o in self.store.pending_orders(account):
                q = self.feed.quote(o["symbol"])
                if q is None:
                    continue  # stays pending → filled from klines later
                self.broker.fill(o, q.mid, now, quote=q)

    # --------------------------------------------------------------- cycle
    def cycle(self, now: pd.Timestamp | None = None) -> CycleReport:
        now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
        self.feed.refresh()
        rep = CycleReport(now=now, mode=self.feed.mode)
        self.ensure_accounts(now)
        horizon = min(self.feed.horizon(self.cfg.universe.symbols), now.floor("h"))
        rep.horizon = horizon
        with self.store.transaction():
            for spec in self.specs:
                self._advance(spec.name, horizon, rep)
        if self.feed.mode == "realtime":
            for spec in self.specs:
                if spec.vault is not None:
                    self._maybe_bank(spec, now, rep)

        staleness = (now - horizon) / pd.Timedelta(hours=1)
        bar_ts = latest_bar(horizon, self.cfg.market.bar)
        rep.signal_bar = bar_ts
        if staleness > self.cfg.risk.stale_data_hours:
            msg = f"data stale by {staleness:.1f}h (> {self.cfg.risk.stale_data_hours}h); not trading"
            rep.warnings.append(msg)
            with self.store.transaction():
                self.store.log_event(now, "warning", msg)
            self._export(rep)
            return rep
        if now < bar_ts + pd.Timedelta(hours=self.cfg.market.exec_lag_hours):
            self._export(rep)
            return rep

        if self.halt_file.exists():
            rep.warnings.append("HALT file present: flattening all accounts, no new risk")
        due = []
        for spec in self.specs:
            last = self.store.last_decision(spec.name)
            if last is not None and utc(last["bar_ts"]) >= bar_ts:
                continue
            if self.store.pending_orders(spec.name):
                rep.warnings.append(f"{spec.name}: pending orders, decision deferred")
                continue
            due.append(spec)
        if due:
            # Only hourly bars that closed by the signal bar's close are used.
            panel = self.feed.panel(self.cfg.market.bar, self.cfg.market.exec_lag_hours, end=bar_ts)
            panel = panel.slice(None, bar_ts)
            if panel.close.index[-1] != bar_ts:
                rep.warnings.append(f"panel ends {panel.close.index[-1]} != signal bar {bar_ts}")
            else:
                cache: dict = {}
                for spec in due:
                    try:
                        self._decide(spec, panel, bar_ts, horizon, now, cache, rep)
                    except Exception as exc:  # noqa: BLE001 - one bad strategy must not stop the others
                        log.exception("decision failed for %s", spec.name)
                        rep.warnings.append(f"{spec.name}: decision failed: {exc}")
                        with self.store.transaction():
                            self.store.log_event(now, "error", f"decision failed: {exc}", spec.name)
        self._export(rep)
        return rep

    # --------------------------------------------------------------- vault
    def _quotes_for(self, account: str, fallback_ts: pd.Timestamp | None = None) -> dict | None:
        """Live quotes for every open position (None if one is missing); with ``fallback_ts`` a missing
        quote is replaced by the last known price at that time (used only to close out at the end)."""
        from tradinglab.data.exchanges import Quote

        out = {}
        for s in self.store.positions(account):
            q = self.feed.quote(s)
            if q is None and fallback_ts is not None:
                px = self.feed.price_at(s, fallback_ts)
                if px is not None:
                    log.warning("%s: no live quote for %s, using last price %.6g", account, s, px)
                    q = Quote(px, px, fallback_ts, "last-price")
            if q is None:
                return None
            out[s] = q
        return out

    def _close_all(self, account: str, quotes: dict, now: pd.Timestamp, reason: str) -> int:
        """Close every position at ``quotes`` (caller holds the transaction)."""
        ids = self.broker.rebalance(account, {}, {s: q.mid for s, q in quotes.items()}, now, now, 0.0, 0.0,
                                    reason=reason)
        for o in self.store.pending_orders(account):
            q = quotes[o["symbol"]]
            self.broker.fill(o, q.mid, now, quote=q)
        return len(ids)

    def _maybe_bank(self, spec: AccountSpec, now: pd.Timestamp, rep: CycleReport) -> None:
        """Vault rule, evaluated once per hourly mark: realise everything, bank the excess over the base.
        Closing fills, the transfer and the bookkeeping commit together (one transaction)."""
        account = spec.name
        last = self.store.last_equity(account)
        if last is None or self.store.get_meta(f"vault_checked:{account}") == last["ts"]:
            return
        if self.store.pending_orders(account):
            return  # wait until the book is settled
        base = self.store.account(account)["initial_capital"]
        due = spec.vault.due(last["equity"], base)
        quotes = self._quotes_for(account) if due else {}
        if quotes is None:
            rep.warnings.append(f"{account}: vault due but a quote is missing; retry next cycle")
            return
        with self.store.transaction():
            if due:
                n = self._close_all(account, quotes, now, f"vault: realise profit (mark {last['ts']})")
                amount = bank(self.store, account, now, last["equity"],
                              note=f"threshold {spec.vault.threshold:g}", mark_ts=last["ts"])
                self.store.set_meta(f"guard:{account}", {})
                rep.orders += n
                rep.banked += amount
            self.store.set_meta(f"vault_checked:{account}", last["ts"])

    def liquidate(self, now: pd.Timestamp | None = None, reason: str = "experiment end") -> dict[str, float]:
        """Close every account at live quotes (last known price if a coin has none) and mark the final
        equity (= cash)."""
        now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
        self.feed.refresh()
        horizon = self.feed.horizon(self.cfg.universe.symbols)
        out = {}
        for spec in self.specs:
            if self.store.account(spec.name) is None:
                continue
            quotes = self._quotes_for(spec.name, fallback_ts=horizon)
            if quotes is None:
                raise RuntimeError(f"{spec.name}: cannot liquidate, no price for an open position")
            with self.store.transaction():
                for o in self.store.pending_orders(spec.name):
                    self.store.cancel_order(o["id"], reason)
                self._close_all(spec.name, quotes, now, reason)
                cash = self.store.account(spec.name)["cash"]
                self.store.record_equity(spec.name, now, cash, cash, 0.0, 0.0)
                self.store.log_event(now, "info", f"liquidated ({reason})", spec.name)
            out[spec.name] = cash
        self._export(CycleReport(now=now, mode=self.feed.mode, horizon=horizon,
                                 warnings=[f"liquidated: {reason}"]))
        return out

    # -------------------------------------------------------------- export
    def status(self, horizon: pd.Timestamp | None = None) -> dict:
        out = {"accounts": {}}
        for spec in self.specs:
            acct = self.store.account(spec.name)
            if acct is None:
                continue
            last = self.store.last_equity(spec.name)
            eq = last["equity"] if last else acct["initial_capital"]
            positions = self.store.positions(spec.name)
            weights = {}
            if horizon is not None and eq > 0:
                for sym, units in positions.items():
                    px = self.feed.price_at(sym, horizon)
                    if px is not None:
                        weights[sym] = units * px / eq
            vault = self.store.vault_total(spec.name)
            # Profit banked after the latest mark is still inside that mark's equity: count it once.
            vault_in_pnl = self.store.vault_total(spec.name, before_mark=last["ts"]) if last else vault
            out["accounts"][spec.name] = {
                "strategy": spec.label,
                "vault_threshold": spec.vault.threshold if spec.vault else None,
                "initial_capital": acct["initial_capital"],
                "equity": eq,
                "vault": vault,
                "pnl": eq + vault_in_pnl - acct["initial_capital"],
                "return": (eq + vault_in_pnl) / acct["initial_capital"] - 1.0,
                "marked_at": last["ts"] if last else None,
                "gross": last["gross"] if last else 0.0,
                "net": last["net"] if last else 0.0,
                "positions": positions,
                "weights": weights,
                "pending_orders": len(self.store.pending_orders(spec.name)),
                "funding_paid": -self.store.funding_total(spec.name),
            }
        return out

    def _export(self, rep: CycleReport) -> None:
        state_dir = Path(self.cfg.paths.state_dir)
        status = self.status(rep.horizon)
        status.update({
            "updated_at": str(rep.now), "feed_mode": rep.mode, "data_horizon": str(rep.horizon),
            "signal_bar": str(rep.signal_bar), "last_cycle": {
                "fills": rep.fills, "funding_events": rep.funding_events, "marks": rep.marks,
                "decisions": rep.decisions, "orders": rep.orders, "warnings": rep.warnings},
            "selection": {"strategies": self.selection.get("strategies", []),
                          "alphas": self.selection.get("alphas", {}),
                          "created_at": self.selection.get("created_at")},
        })
        (state_dir / "status.json").write_text(json.dumps(status, indent=2, default=str))
        journal = state_dir / "journal"
        journal.mkdir(exist_ok=True)
        for spec in self.specs:
            safe = spec.name.replace(":", "_")
            eq = self.store.equity_curve(spec.name)
            if not eq.empty:
                eq.to_csv(journal / f"equity_{safe}.csv")
        orders = self.store.orders(limit=100000)
        if not orders.empty:
            orders.sort_values("id").to_csv(journal / "orders.csv", index=False)
