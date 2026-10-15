"""Experiment orchestration: snapshots, records push, close-out failure handling."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import pandas as pd
import pytest

from tradinglab import experiment as X
from tradinglab.config import load_config
from tradinglab.live.state import StateStore


def _repo(tmp_path, branch="experiment/test"):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    rec = tmp_path / "records"
    subprocess.run(["git", "init", "-q", "-b", branch, str(rec)], check=True)
    for k, v in (("user.name", "t"), ("user.email", "t@x")):
        subprocess.run(["git", "-C", str(rec), "config", k, v], check=True)
    subprocess.run(["git", "-C", str(rec), "remote", "add", "origin", str(remote)], check=True)
    return remote, rec


def _cfg(tmp_path):
    cfg = load_config("config/experiment.yaml")
    cfg.paths.state_dir = str(tmp_path / "run" / "state")
    cfg.experiment.records_branch = "experiment/test"
    return cfg


def test_snapshot_is_consistent_and_pushed(tmp_path):
    remote, rec = _repo(tmp_path)
    cfg = _cfg(tmp_path)
    sd = tmp_path / "run" / "state"
    st = StateStore(sd / "intraday.sqlite")
    now = pd.Timestamp("2026-10-16", tz="UTC")
    st.ensure_account("intraday:a", "x", 100_000.0, now)
    st.record_equity("intraday:a", now, 101_000.0, 101_000.0, 0.0, 0.0)
    st.record_vault("intraday:a", now, 1_000.0, 101_000.0, 101_000.0, mark_ts=now)
    X.experiment_meta(cfg, rec)
    X.snapshot(sd, rec)
    assert X.git_push(rec, "experiment/test", "snap")
    files = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "experiment/test"],
                           capture_output=True, text=True, check=True).stdout.split()
    assert {"state/intraday.sqlite", "csv/intraday_vault.csv", "csv/intraday_equity.csv", "experiment.json",
            "heartbeat.json"} <= set(files)
    con = sqlite3.connect(rec / "state" / "intraday.sqlite")
    assert con.execute("select amount from vault").fetchone()[0] == 1_000.0
    # a fresh runner restores exactly what was committed
    fresh = tmp_path / "fresh"
    assert X.restore(fresh, rec) == ["intraday.sqlite"]
    assert StateStore(fresh / "intraday.sqlite").vault_total("intraday:a") == 1_000.0


def test_failed_close_out_pushes_records_and_is_retried(tmp_path, monkeypatch):
    remote, rec = _repo(tmp_path)
    cfg = _cfg(tmp_path)
    rec.mkdir(exist_ok=True)
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=15)
    (rec / "experiment.json").write_text(json.dumps({"name": "t", "start": str(start),
                                                     "end": str(start + pd.Timedelta(days=14))}))
    calls = {"n": 0}

    def failing_finish(*a, **k):
        calls["n"] += 1
        raise ConnectionError("exchange down")

    monkeypatch.setattr(X, "finish", failing_finish)
    assert X.run(cfg, None, rec, minutes=5) == "finish-failed"     # over already: no daemons are started
    assert (rec / "FINISH_ERROR.txt").exists() and not (rec / "FINISHED.json").exists()
    log = subprocess.run(["git", "--git-dir", str(remote), "log", "--format=%s", "experiment/test"],
                         capture_output=True, text=True, check=True).stdout
    assert "finish-failed" in log and "final snapshot before close-out" in log

    def ok_finish(cfg_, records, meta):
        calls["n"] += 1
        (records / "FINISHED.json").write_text("{}")
        return {}

    monkeypatch.setattr(X, "finish", ok_finish)
    assert X.run(cfg, None, rec, minutes=5) == "finished"
    assert not (rec / "FINISH_ERROR.txt").exists()
    assert X.run(cfg, None, rec, minutes=5) == "already-finished"
    assert calls["n"] == 2
