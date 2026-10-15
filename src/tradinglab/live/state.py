"""SQLite persistence for paper-trading accounts.

Everything needed to resume after a crash or a container restart lives in one
file (``state/lab.sqlite``): balances, positions, every order and fill, the
hourly equity curve, funding payments, decisions and an event journal. All
writes for one engine cycle happen in a single transaction.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    name TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    initial_capital REAL NOT NULL,
    created_at TEXT NOT NULL,
    strategy TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    account TEXT NOT NULL,
    symbol TEXT NOT NULL,
    units REAL NOT NULL,
    PRIMARY KEY (account, symbol)
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL,
    symbol TEXT NOT NULL,
    qty REAL NOT NULL,
    submit_ts TEXT NOT NULL,
    fill_after_ts TEXT NOT NULL,
    ref_price REAL,
    status TEXT NOT NULL,           -- pending | filled | cancelled
    fill_ts TEXT,
    fill_price REAL,
    fee REAL,
    slippage_cost REAL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS orders_pending ON orders(account, status);
CREATE TABLE IF NOT EXISTS equity (
    account TEXT NOT NULL,
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    gross REAL NOT NULL,
    net REAL NOT NULL,
    PRIMARY KEY (account, ts)
);
CREATE TABLE IF NOT EXISTS funding (
    account TEXT NOT NULL,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    rate REAL NOT NULL,
    amount REAL NOT NULL,
    estimated INTEGER NOT NULL,
    PRIMARY KEY (account, ts, symbol)
);
CREATE TABLE IF NOT EXISTS decisions (
    account TEXT NOT NULL,
    bar_ts TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    equity_estimate REAL,
    risk_scale REAL,
    targets TEXT NOT NULL,
    PRIMARY KEY (account, bar_ts)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vault (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL,
    ts TEXT NOT NULL,
    amount REAL NOT NULL,            -- realised profit moved out of the trading account
    equity_marked REAL NOT NULL,     -- equity at the mark that triggered the sweep
    equity_realised REAL NOT NULL,   -- cash after every position was closed
    note TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    account TEXT,
    message TEXT NOT NULL
);
"""


def ts_str(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").isoformat() if pd.Timestamp(ts).tzinfo else \
        pd.Timestamp(ts).tz_localize("UTC").isoformat()


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=DELETE")  # single file → git friendly
        self.conn.executescript(SCHEMA)
        self._depth = 0

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self):
        if self._depth == 0:
            self.conn.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.conn.execute("ROLLBACK")
            raise
        else:
            self._depth -= 1
            if self._depth == 0:
                self.conn.execute("COMMIT")

    # ----------------------------------------------------------------- meta
    def get_meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_meta(self, key: str, value) -> None:
        self.conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (key, json.dumps(value, default=str)))

    def log_event(self, ts, level: str, message: str, account: str | None = None) -> None:
        self.conn.execute("INSERT INTO events(ts, level, account, message) VALUES (?,?,?,?)",
                          (ts_str(ts), level, account, message))

    # ------------------------------------------------------------- accounts
    def ensure_account(self, name: str, strategy: str, capital: float, now) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO accounts(name, cash, initial_capital, created_at, strategy) VALUES (?,?,?,?,?)",
            (name, capital, capital, ts_str(now), strategy))

    def accounts(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM accounts ORDER BY name")]

    def account(self, name: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM accounts WHERE name=?", (name,)).fetchone()
        return dict(r) if r else None

    def set_cash(self, name: str, cash: float) -> None:
        self.conn.execute("UPDATE accounts SET cash=? WHERE name=?", (cash, name))

    def positions(self, account: str) -> dict[str, float]:
        return {r["symbol"]: r["units"] for r in
                self.conn.execute("SELECT symbol, units FROM positions WHERE account=?", (account,))}

    def set_position(self, account: str, symbol: str, units: float) -> None:
        if abs(units) < 1e-12:
            self.conn.execute("DELETE FROM positions WHERE account=? AND symbol=?", (account, symbol))
        else:
            self.conn.execute(
                "INSERT INTO positions(account, symbol, units) VALUES (?,?,?) "
                "ON CONFLICT(account, symbol) DO UPDATE SET units=excluded.units", (account, symbol, units))

    # --------------------------------------------------------------- orders
    def add_order(self, account: str, symbol: str, qty: float, submit_ts, fill_after_ts,
                  ref_price: float, reason: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO orders(account, symbol, qty, submit_ts, fill_after_ts, ref_price, status, reason) "
            "VALUES (?,?,?,?,?,?, 'pending', ?)",
            (account, symbol, qty, ts_str(submit_ts), ts_str(fill_after_ts), ref_price, reason))
        return int(cur.lastrowid)

    def pending_orders(self, account: str | None = None) -> list[dict]:
        if account is None:
            rows = self.conn.execute("SELECT * FROM orders WHERE status='pending' ORDER BY id")
        else:
            rows = self.conn.execute("SELECT * FROM orders WHERE status='pending' AND account=? ORDER BY id",
                                     (account,))
        return [dict(r) for r in rows]

    def fill_order(self, order_id: int, fill_ts, price: float, fee: float, slippage_cost: float) -> None:
        self.conn.execute("UPDATE orders SET status='filled', fill_ts=?, fill_price=?, fee=?, slippage_cost=? "
                          "WHERE id=?", (ts_str(fill_ts), price, fee, slippage_cost, order_id))

    def cancel_order(self, order_id: int, reason: str) -> None:
        self.conn.execute("UPDATE orders SET status='cancelled', reason=reason || ' | ' || ? WHERE id=?",
                          (reason, order_id))

    def orders(self, account: str | None = None, limit: int = 1000) -> pd.DataFrame:
        q = "SELECT * FROM orders" + (" WHERE account=?" if account else "") + " ORDER BY id DESC LIMIT ?"
        args = (account, limit) if account else (limit,)
        return pd.DataFrame([dict(r) for r in self.conn.execute(q, args)])

    # --------------------------------------------------------------- equity
    def record_equity(self, account: str, ts, equity: float, cash: float, gross: float, net: float) -> None:
        self.conn.execute(
            "INSERT INTO equity(account, ts, equity, cash, gross, net) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(account, ts) DO UPDATE SET equity=excluded.equity, cash=excluded.cash, "
            "gross=excluded.gross, net=excluded.net", (account, ts_str(ts), equity, cash, gross, net))

    def equity_curve(self, account: str) -> pd.DataFrame:
        df = pd.DataFrame([dict(r) for r in self.conn.execute(
            "SELECT ts, equity, cash, gross, net FROM equity WHERE account=? ORDER BY ts", (account,))])
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.set_index("ts")
        return df

    def last_equity(self, account: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM equity WHERE account=? ORDER BY ts DESC LIMIT 1", (account,)).fetchone()
        return dict(r) if r else None

    def record_funding(self, account: str, ts, symbol: str, rate: float, amount: float, estimated: bool) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO funding(account, ts, symbol, rate, amount, estimated) VALUES (?,?,?,?,?,?)",
            (account, ts_str(ts), symbol, rate, amount, int(estimated)))

    def funding_total(self, account: str) -> float:
        r = self.conn.execute("SELECT COALESCE(SUM(amount), 0) AS s FROM funding WHERE account=?", (account,)).fetchone()
        return float(r["s"])

    # ------------------------------------------------------------ decisions
    def record_decision(self, account: str, bar_ts, decided_at, equity_estimate: float, risk_scale: float,
                        targets: dict[str, float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO decisions(account, bar_ts, decided_at, equity_estimate, risk_scale, targets) "
            "VALUES (?,?,?,?,?,?)",
            (account, ts_str(bar_ts), ts_str(decided_at), equity_estimate, risk_scale, json.dumps(targets)))

    def last_decision(self, account: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM decisions WHERE account=? ORDER BY bar_ts DESC LIMIT 1",
                              (account,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["targets"] = json.loads(d["targets"])
        return d

    # ---------------------------------------------------------------- vault
    def record_vault(self, account: str, ts, amount: float, equity_marked: float, equity_realised: float,
                     note: str = "") -> None:
        self.conn.execute("INSERT INTO vault(account, ts, amount, equity_marked, equity_realised, note) "
                          "VALUES (?,?,?,?,?,?)", (account, ts_str(ts), amount, equity_marked, equity_realised, note))

    def vault_total(self, account: str) -> float:
        r = self.conn.execute("SELECT COALESCE(SUM(amount), 0) AS s FROM vault WHERE account=?", (account,)).fetchone()
        return float(r["s"])

    def vault_events(self, account: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM vault" + (" WHERE account=?" if account else "") + " ORDER BY id"
        return pd.DataFrame([dict(r) for r in self.conn.execute(q, (account,) if account else ())])

    def events(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]
