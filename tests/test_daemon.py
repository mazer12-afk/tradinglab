import json
import sys

import pandas as pd

from tradinglab.live.daemon import Daemon


def _daemon(cfg, code: int):
    d = Daemon(cfg, research_days=30)
    d._cli = lambda *args: [sys.executable, "-c", f"import sys; sys.exit({code})"]
    return d


def test_failed_research_is_retried_after_cooloff(cfg):
    d = _daemon(cfg, 1)
    now = pd.Timestamp("2026-01-01", tz="UTC")
    d.maybe_maintain(now)                      # no selection yet → runs synchronously, fails
    m = json.loads(d.maint_path.read_text())
    assert "research" not in m and pd.Timestamp(m["retry_after"]) == now + pd.Timedelta(hours=6)
    d.maybe_maintain(now + pd.Timedelta(hours=1))
    assert d._child is None                    # still cooling off


def test_successful_research_schedules_next_run(cfg, tmp_path):
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / "selection.json").write_text("{}")
    d = _daemon(cfg, 0)
    now = pd.Timestamp("2026-01-01", tz="UTC")
    d.maybe_maintain(now)                      # selection exists → runs in the background
    d._child.wait()
    d.maybe_maintain(now + pd.Timedelta(minutes=5))
    m = json.loads(d.maint_path.read_text())
    assert pd.Timestamp(m["research"]) == now and d._child is None
    d.maybe_maintain(now + pd.Timedelta(days=8))   # weekly retrain due
    assert d._child is not None and d._child_key == "retrain"
    d._child.wait()
