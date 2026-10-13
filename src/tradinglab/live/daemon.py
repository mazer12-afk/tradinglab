"""Long-running autonomous loop: trade, maintain models, never die.

* runs :meth:`LiveEngine.cycle` every ``live.poll_seconds``;
* retrains the live deep models every ``live.retrain_days`` and re-runs the
  full research tournament every ``research_days`` in a *subprocess* (so a
  crash or memory spike there cannot take the trading loop down);
* survives any exception with exponential back-off and writes a heartbeat
  file for external watchdogs (systemd, Docker healthcheck, k8s probe).
"""

from __future__ import annotations

import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from tradinglab.config import LabConfig, utc

log = logging.getLogger(__name__)


class Daemon:
    def __init__(self, cfg: LabConfig, config_path: str | None = None, research_days: int = 30,
                 max_cycles: int | None = None, git_commit: bool = False):
        self.cfg = cfg
        self.config_path = config_path
        self.research_days = research_days
        self.max_cycles = max_cycles
        self.git_commit = git_commit
        self.stop = False
        self.state_dir = Path(cfg.paths.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.maint_path = self.state_dir / "maintenance.json"
        self._child: subprocess.Popen | None = None

    def _handle_signal(self, signum, _frame):
        log.info("signal %s received – stopping after current step", signum)
        self.stop = True

    def heartbeat(self, status: str, **extra) -> None:
        payload = {"ts": pd.Timestamp.now(tz="UTC").isoformat(), "status": status, **extra}
        (self.state_dir / "heartbeat.json").write_text(json.dumps(payload, default=str))

    # --------------------------------------------------------- maintenance
    def _maint(self) -> dict:
        return json.loads(self.maint_path.read_text()) if self.maint_path.exists() else {}

    def _cli(self, *args: str) -> list[str]:
        cmd = [sys.executable, "-m", "tradinglab.cli"]
        if self.config_path:
            cmd += ["--config", self.config_path]
        return cmd + list(args)

    def maybe_maintain(self, now: pd.Timestamp) -> None:
        if self._child is not None:
            rc = self._child.poll()
            if rc is None:
                return
            log.info("maintenance job finished with code %s", rc)
            self._child = None
        m = self._maint()
        last_research = utc(m["research"]) if m.get("research") else None
        last_retrain = utc(m["retrain"]) if m.get("retrain") else None
        job = None
        if last_research is None or now - last_research > pd.Timedelta(days=self.research_days):
            job, key = self._cli("research"), "research"
        elif last_retrain is None or now - last_retrain > pd.Timedelta(days=self.cfg.live.retrain_days):
            job, key = self._cli("train-live"), "retrain"
        if job is None:
            return
        # Bootstrap: without any selection yet, research must finish before trading.
        sel_exists = (Path(self.cfg.paths.models_dir) / "selection.json").exists()
        log.info("starting maintenance job: %s", " ".join(job))
        m[key] = now.isoformat()
        if key == "research":
            m["retrain"] = now.isoformat()
        self.maint_path.write_text(json.dumps(m))
        log_file = open(self.state_dir / f"{key}.log", "a")  # noqa: SIM115 - handed to the child
        self._child = subprocess.Popen(job, stdout=log_file, stderr=subprocess.STDOUT)
        if not sel_exists:
            self._child.wait()
            self._child = None

    # ---------------------------------------------------------------- loop
    def run(self) -> None:
        from tradinglab.data.store import MarketData
        from tradinglab.live.engine import LiveEngine, load_selection
        from tradinglab.live.feed import LiveFeed
        from tradinglab.live.state import StateStore

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        store = StateStore(self.state_dir / "lab.sqlite")
        md = MarketData(self.cfg.paths.data_dir)
        feed = LiveFeed(md, self.cfg.universe.symbols, self.cfg.walkforward.data_start, self.cfg.live.feed,
                        default_funding_8h=self.cfg.market.default_funding_8h)
        backoff = 30.0
        cycles = 0
        selection_mtime = None
        engine = None
        while not self.stop:
            now = pd.Timestamp.now(tz="UTC")
            try:
                self.maybe_maintain(now)
                sel_path = Path(self.cfg.paths.models_dir) / "selection.json"
                mtime = sel_path.stat().st_mtime if sel_path.exists() else None
                if engine is None or mtime != selection_mtime:
                    engine = LiveEngine(self.cfg, feed, store, load_selection(self.cfg))
                    selection_mtime = mtime
                    log.info("engine loaded selection: %s", engine.selection.get("strategies"))
                rep = engine.cycle(now)
                log.info("cycle: mode=%s horizon=%s fills=%d marks=%d decisions=%s orders=%d warnings=%s",
                         rep.mode, rep.horizon, rep.fills, rep.marks, rep.decisions, rep.orders, rep.warnings)
                self.heartbeat("ok", mode=rep.mode, horizon=str(rep.horizon))
                if self.git_commit and (rep.decisions or rep.fills):
                    commit_state(self.state_dir, f"live: {rep.now:%Y-%m-%d %H:%M} decisions={len(rep.decisions)} "
                                                 f"fills={rep.fills}")
                backoff = 30.0
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                log.exception("cycle failed")
                self.heartbeat("error", error=str(exc))
                time.sleep(min(backoff, 3600))
                backoff *= 2
                continue
            cycles += 1
            if self.max_cycles is not None and cycles >= self.max_cycles:
                break
            for _ in range(int(self.cfg.live.poll_seconds)):
                if self.stop:
                    break
                time.sleep(1)
        store.close()
        self.heartbeat("stopped")


def commit_state(state_dir: Path, message: str) -> None:
    """Commit the state directory to git (audit trail); failures are logged, not raised."""
    try:
        subprocess.run(["git", "add", str(state_dir)], check=True, capture_output=True)
        r = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["git", "push"], capture_output=True, timeout=120)
    except Exception as exc:  # noqa: BLE001
        log.warning("git commit of state failed: %s", exc)
