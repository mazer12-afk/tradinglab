"""Unattended, fixed-length live experiment with records committed to git.

One ``run`` is one shift of a few hours (a GitHub Actions job lasts at most 6 h;
the workflow chains shifts back to back). A shift:

1. restores the runtime state (``cfg.paths.state_dir``) from the records
   checkout – the records branch *is* the experiment's persistent memory;
2. supervises the two trading daemons (intraday 5-minute AI accounts, daily
   strategy accounts) as subprocesses, restarting either if it dies;
3. every ``commit_every`` minutes writes a consistent snapshot – SQLite copies
   via the backup API, human-readable CSV exports, status JSON, logs – into the
   records checkout and pushes it;
4. stops at its deadline; if the experiment's end has been reached instead,
   liquidates every account at live prices, writes ``FINISHED.json`` and stops
   for good.

Bars missed between shifts are replayed on restart (marked, never traded), so
nothing is skipped and no trade is made on stale information.
"""

from __future__ import annotations

import json
import logging
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from tradinglab.config import LabConfig, utc

log = logging.getLogger(__name__)

DBS = ("intraday.sqlite", "lab.sqlite")
TABLES = {  # table → ORDER BY (time first, so new rows append at the end of each CSV)
    "equity": "ts, account", "orders": "id", "vault": "id", "funding": "ts, account, symbol",
    "decisions": "decided_at, account", "accounts": "name", "events": "id",
}


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


# ---------------------------------------------------------------- calendar
def experiment_meta(cfg: LabConfig, records: Path, create: bool = True) -> dict:
    """Start/end of the experiment; the start is fixed by the first run and kept in the records."""
    path = records / "experiment.json"
    if path.exists():
        meta = json.loads(path.read_text())
    else:
        start = now_utc()
        end = utc(cfg.experiment.end) if cfg.experiment.end else start + pd.Timedelta(days=cfg.experiment.duration_days)
        meta = {"name": cfg.experiment.name, "start": str(start), "end": str(end),
                "vault_thresholds": {"intraday": cfg.intraday.vault_thresholds, "daily": cfg.live.vault_thresholds},
                "intraday_policies": cfg.intraday.live_policies,
                "protocol": "frozen models and rules; no parameter changes and no performance analysis until the "
                            "end; every account starts with its initial capital; vault twins bank realised profit "
                            "above the initial capital once equity reaches it·(1+threshold) and restart from it."}
        if create:
            records.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------- snapshots
def restore(state_dir: Path, records: Path) -> list[str]:
    """Copy the last committed databases into the runtime state dir (fresh runner)."""
    state_dir.mkdir(parents=True, exist_ok=True)
    restored = []
    for db in DBS:
        src, dst = records / "state" / db, state_dir / db
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            restored.append(db)
    return restored


def _backup(src: Path, dst: Path) -> None:
    """Consistent copy of a live SQLite database (safe while a daemon writes to it)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)
    d = sqlite3.connect(tmp)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    tmp.replace(dst)


def snapshot(state_dir: Path, records: Path, logs: dict[str, Path] | None = None) -> None:
    """Write databases, CSV exports, status files and logs into the records checkout."""
    for db in DBS:
        src = state_dir / db
        if not src.exists():
            continue
        dst = records / "state" / db
        _backup(src, dst)
        prefix = "intraday" if db.startswith("intraday") else "daily"
        con = sqlite3.connect(dst)
        try:
            for table, order in TABLES.items():
                df = pd.read_sql(f"SELECT * FROM {table} ORDER BY {order}", con)
                out = records / "csv" / f"{prefix}_{table}.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False)
        finally:
            con.close()
    for name, dst in (("intraday_status.json", "status/intraday_status.json"),
                      ("status.json", "status/daily_status.json")):
        if (state_dir / name).exists():
            (records / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(state_dir / name, records / dst)
    for key, path in (logs or {}).items():
        if path.exists():
            (records / "logs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, records / "logs" / f"{key}.log")
    (records / "heartbeat.json").write_text(json.dumps({"ts": str(now_utc())}))


def git_push(records: Path, branch: str, message: str) -> bool:
    """Commit everything in the records checkout and push it; never raises."""
    def git(*args, check=False):
        return subprocess.run(["git", "-C", str(records), *args], capture_output=True, text=True, check=check)

    try:
        git("add", "-A")
        if git("diff", "--cached", "--quiet").returncode == 0:
            return True
        r = git("commit", "-q", "-m", message)
        if r.returncode != 0:
            log.warning("git commit failed: %s", r.stderr.strip())
            return False
        for attempt in range(5):
            r = git("push", "-q", "origin", f"HEAD:{branch}")
            if r.returncode == 0:
                return True
            log.warning("git push failed (attempt %d): %s", attempt + 1, r.stderr.strip())
            git("pull", "-q", "--rebase", "origin", branch)
            time.sleep(2 ** attempt)
    except Exception as exc:  # noqa: BLE001 - recording must never stop trading
        log.warning("git push of records failed: %s", exc)
    return False


# --------------------------------------------------------------- supervisor
class Supervisor:
    """Runs the two daemons until ``deadline``, restarting them if they die."""

    def __init__(self, cfg_path: str | None, deadline: pd.Timestamp, logs_dir: Path):
        self.cfg_path = cfg_path
        self.deadline = deadline
        self.logs_dir = logs_dir
        self.procs: dict[str, subprocess.Popen] = {}
        self.restarts: dict[str, int] = {"intraday": 0, "daily": 0}
        self.logs = {k: logs_dir / f"{k}.log" for k in self.restarts}

    def _cmd(self, which: str) -> list[str]:
        cmd = [sys.executable, "-m", "tradinglab.cli"]
        if self.cfg_path:
            cmd += ["--config", self.cfg_path]
        until = self.deadline.isoformat()
        if which == "intraday":
            return cmd + ["intraday-run", "--poll", "10", "--until", until]
        return cmd + ["run", "--no-maintenance", "--until", until]

    def start(self, which: str) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        fh = open(self.logs[which], "a")  # noqa: SIM115 - owned by the child
        self.procs[which] = subprocess.Popen(self._cmd(which), stdout=fh, stderr=subprocess.STDOUT)
        log.info("started %s daemon (pid %d)", which, self.procs[which].pid)

    def check(self) -> None:
        for which, p in list(self.procs.items()):
            if p.poll() is not None and now_utc() < self.deadline - pd.Timedelta(minutes=2):
                self.restarts[which] += 1
                log.warning("%s daemon exited with %s; restarting (#%d)", which, p.returncode, self.restarts[which])
                time.sleep(min(60, 5 * self.restarts[which]))
                self.start(which)

    def stop(self, grace: float = 240.0) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        t0 = time.time()
        for p in self.procs.values():
            try:
                p.wait(timeout=max(1.0, grace - (time.time() - t0)))
            except subprocess.TimeoutExpired:
                p.kill()


def finish(cfg: LabConfig, records: Path, meta: dict) -> dict:
    """End of experiment: liquidate every account at live prices, write FINISHED.json."""
    from tradinglab.data.store import MarketData
    from tradinglab.intraday.live import IntradayEngine, IntradayFeed
    from tradinglab.intraday.policy import policies_by_name
    from tradinglab.live.engine import LiveEngine, load_selection
    from tradinglab.live.feed import LiveFeed
    from tradinglab.live.state import StateStore

    state_dir = Path(cfg.paths.state_dir)
    end = utc(meta["end"])
    final: dict[str, dict] = {}
    md = MarketData(cfg.paths.data_dir)
    if (state_dir / "intraday.sqlite").exists():
        store = StateStore(state_dir / "intraday.sqlite")
        feed = IntradayFeed(md, cfg.intraday.symbols)
        feed.load()
        eng = IntradayEngine(cfg, feed, store, Path(cfg.paths.models_dir) / "intraday" / "live",
                             policies=policies_by_name(cfg.intraday.live_policies))
        eng.cycle()               # mark the last bars before closing out
        eng.liquidate(reason="experiment end")
        for name, st in eng.status().items():
            final[name] = {k: st[k] for k in ("equity", "vault", "pnl", "return", "sweeps", "trades")}
        store.close()
    if (state_dir / "lab.sqlite").exists():
        store = StateStore(state_dir / "lab.sqlite")
        feed = LiveFeed(md, cfg.universe.symbols, cfg.walkforward.data_start, cfg.live.feed,
                        default_funding_8h=cfg.market.default_funding_8h)
        eng = LiveEngine(cfg, feed, store, load_selection(cfg))
        eng.cycle()
        eng.liquidate(reason="experiment end")
        for name, st in eng.status()["accounts"].items():
            final[f"daily:{name}"] = {k: st[k] for k in ("equity", "vault", "pnl", "return")}
        store.close()
    out = {"finished_at": str(now_utc()), "end": str(end), "start": meta["start"], "final": final}
    (records / "FINISHED.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def run(cfg: LabConfig, cfg_path: str | None, records: Path, minutes: float, commit_every: float = 60.0) -> str:
    """One shift of the experiment. Returns "finished", "already-finished" or "shift-done"."""
    records = Path(records)
    branch = cfg.experiment.records_branch
    if (records / "FINISHED.json").exists():
        log.info("experiment already finished")
        return "already-finished"
    meta = experiment_meta(cfg, records)
    end = utc(meta["end"])
    started = now_utc()
    deadline = min(started + pd.Timedelta(minutes=minutes), end)
    state_dir = Path(cfg.paths.state_dir)
    restored = restore(state_dir, records)
    run_id = f"{started:%Y%m%dT%H%M}"
    logs_dir = state_dir.parent / "logs"
    with open(records / "runs.csv", "a") as fh:
        if fh.tell() == 0:
            fh.write("run_id,started,deadline,restored\n")
        fh.write(f"{run_id},{started},{deadline},{'+'.join(restored)}\n")
    git_push(records, branch, f"experiment {meta['name']}: shift {run_id} started")
    log.info("shift %s: until %s (experiment end %s); restored %s", run_id, deadline, end, restored or "nothing")

    sup = Supervisor(cfg_path, deadline, logs_dir)
    sup.start("intraday")
    sup.start("daily")
    named_logs = {f"{run_id}_{k}": p for k, p in sup.logs.items()}
    last_commit = time.time()
    try:
        while now_utc() < deadline:
            time.sleep(20)
            sup.check()
            if time.time() - last_commit >= commit_every * 60:
                snapshot(state_dir, records, named_logs)
                git_push(records, branch, f"experiment {meta['name']}: {now_utc():%Y-%m-%d %H:%M} UTC")
                last_commit = time.time()
    finally:
        sup.stop()
    snapshot(state_dir, records, named_logs)
    status = "shift-done"
    if now_utc() >= end:
        finish(cfg, records, meta)
        snapshot(state_dir, records, named_logs)
        status = "finished"
    git_push(records, branch, f"experiment {meta['name']}: shift {run_id} {status} "
                              f"(restarts intraday={sup.restarts['intraday']} daily={sup.restarts['daily']})")
    return status
