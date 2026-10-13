"""Paper broker: unit-level accounting identical in spirit to the backtester.

* Buy fills at ``price·(1+slippage)``, sells at ``price·(1-slippage)``; a fee of
  ``fee_bps`` is charged on filled notional. With a live quote the fill starts
  from the touch (ask for buys, bid for sells) and adds the slippage as impact.
* Positions are linear (perpetual-style) units; cash may go negative for
  leveraged longs and above equity for shorts – equity = cash + Σ units·price.
* Funding: at each funding stamp ``cash -= units · price · rate``.
"""

from __future__ import annotations

import logging

import pandas as pd

from tradinglab.config import CostConfig
from tradinglab.live.state import StateStore

log = logging.getLogger(__name__)


class PaperBroker:
    def __init__(self, store: StateStore, costs: CostConfig, instrument: str = "perp"):
        self.store = store
        self.costs = costs
        self.instrument = instrument

    # ----------------------------------------------------------- valuation
    def equity(self, account: str, prices: dict[str, float]) -> tuple[float, float, float]:
        """(equity, gross exposure, net exposure) using ``prices``; missing prices count 0 exposure."""
        acct = self.store.account(account)
        cash = acct["cash"]
        pos = self.store.positions(account)
        mv = {s: u * prices[s] for s, u in pos.items() if prices.get(s) is not None}
        eq = cash + sum(mv.values())
        gross = sum(abs(v) for v in mv.values())
        net = sum(mv.values())
        return eq, gross, net

    # -------------------------------------------------------------- orders
    def submit(self, account: str, symbol: str, qty: float, now, fill_after, ref_price: float,
               reason: str) -> int:
        if self.instrument == "spot":
            held = self.store.positions(account).get(symbol, 0.0)
            qty = max(qty, -held)  # never go short on spot
        return self.store.add_order(account, symbol, qty, now, fill_after, ref_price, reason)

    def fill(self, order: dict, market_price: float, ts, quote=None) -> None:
        qty = order["qty"]
        side = 1.0 if qty > 0 else -1.0
        slip = self.costs.slippage_bps * 1e-4
        if quote is not None:
            # Cross the live spread, plus impact beyond the touch.
            px = (quote.ask if qty > 0 else quote.bid) * (1.0 + side * slip)
        else:
            px = market_price * (1.0 + side * slip)
        notional = abs(qty) * px
        fee = notional * self.costs.fee_bps * 1e-4
        slippage_cost = abs(qty) * abs(px - market_price)
        acct = self.store.account(order["account"])
        self.store.set_cash(order["account"], acct["cash"] - qty * px - fee)
        units = self.store.positions(order["account"]).get(order["symbol"], 0.0) + qty
        self.store.set_position(order["account"], order["symbol"], units)
        self.store.fill_order(order["id"], ts, px, fee, slippage_cost)

    def apply_funding(self, account: str, symbol: str, ts, rate: float, price: float, estimated: bool) -> float:
        if self.instrument != "perp":
            return 0.0
        units = self.store.positions(account).get(symbol, 0.0)
        if units == 0.0:
            return 0.0
        amount = -units * price * rate
        acct = self.store.account(account)
        self.store.set_cash(account, acct["cash"] + amount)
        self.store.record_funding(account, ts, symbol, rate, amount, estimated)
        return amount

    # ---------------------------------------------------------- rebalance
    def rebalance(self, account: str, targets: dict[str, float], prices: dict[str, float], now,
                  fill_after, band: float, min_notional: float, reason: str) -> list[int]:
        eq, _, _ = self.equity(account, prices)
        if eq <= 0:
            log.error("%s: non-positive equity %.2f – refusing to trade", account, eq)
            return []
        pos = self.store.positions(account)
        ids = []
        for sym in sorted(set(targets) | set(pos)):
            price = prices.get(sym)
            if price is None or price <= 0:
                continue
            w_t = float(targets.get(sym, 0.0))
            units = pos.get(sym, 0.0)
            w_c = units * price / eq
            closing = w_t == 0.0 and units != 0.0
            if abs(w_t - w_c) <= band and not closing:
                continue
            qty = w_t * eq / price - units
            if abs(qty * price) < min_notional and not closing:
                continue
            ids.append(self.submit(account, sym, qty, now, fill_after, price, reason))
        return ids


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")
