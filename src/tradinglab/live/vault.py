"""Profit vault: bank realised profit, then trade on from the original capital.

For an account with base capital ``B`` (its initial capital) and threshold ``θ``:

* when its marked equity reaches ``B·(1+θ)`` every position is closed at live
  prices (spread, impact and fees paid), the realised excess ``cash − B`` is
  moved into the account's vault and cash is reset to ``B``; the strategy then
  starts again from a flat book;
* the vault only ever receives money: it is never traded and never refills an
  account, so an account below ``B`` keeps trading what it has left and banks
  again only once it is ``θ`` above ``B``.

Result of an account = vault total + current equity of the trading account.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradinglab.live.state import StateStore


@dataclass(frozen=True)
class VaultRule:
    threshold: float          # fraction of the base, e.g. 0.02 → bank once equity ≥ 1.02·base

    def __post_init__(self):
        if not self.threshold > 0:
            raise ValueError(f"vault threshold must be > 0, got {self.threshold}")

    def due(self, equity: float, base: float) -> bool:
        return equity >= base * (1.0 + self.threshold)

    @property
    def suffix(self) -> str:
        return f"+vault{100 * self.threshold:g}"


def vault_accounts(base_names: list[str], thresholds: list[float]) -> list[tuple[str, str, VaultRule | None]]:
    """(account name, base name, rule): every base account plus one vault twin per threshold."""
    out: list[tuple[str, str, VaultRule | None]] = []
    rules = [VaultRule(float(t)) for t in thresholds]
    for name in base_names:
        out.append((name, name, None))
        out += [(name + r.suffix, name, r) for r in rules]
    return out


def bank(store: StateStore, account: str, ts, equity_marked: float, note: str = "") -> float:
    """Move the account's cash above its base into the vault (the book must already be flat).

    Returns the amount banked (0 if there is nothing above the base)."""
    if store.positions(account):
        raise RuntimeError(f"{account}: cannot bank profit with open positions")
    acct = store.account(account)
    base = acct["initial_capital"]
    amount = acct["cash"] - base
    if amount <= 0:
        return 0.0
    store.set_cash(account, base)
    store.record_vault(account, ts, amount, equity_marked, acct["cash"], note)
    store.log_event(ts, "info", f"vault: banked {amount:,.2f} (marked equity {equity_marked:,.2f})", account)
    return amount
